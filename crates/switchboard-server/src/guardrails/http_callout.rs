//! HTTP external guardrail callout engine.
//!
//! [`HttpCalloutEngine`] POSTs the [`GuardrailInput`] to a remote HTTP service
//! and maps the JSON response back to a [`GuardrailVerdict`].
//!
//! # Wire format
//!
//! **Request** (POST):
//! ```json
//! {
//!   "messages": [{"role": "user", "content": "..."}],
//!   "model": "claude-sonnet-4-20250514",
//!   "user_id": "alice@example.com",
//!   "team": "platform",
//!   "metadata": {}
//! }
//! ```
//!
//! **Response**:
//! ```json
//! {
//!   "action": "pass" | "block" | "modify" | "audit_log",
//!   "rule": "pii-detection",
//!   "reason": "Found SSN pattern",
//!   "confidence": 0.95,
//!   "modified_content": "..."
//! }
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::config::guardrails::EngineConfig;
use crate::error::ServerError;
use crate::guardrails::engine::{
    AuditSeverity, GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};
use crate::guardrails::pipeline::FailMode;

// ── Wire types ────────────────────────────────────────────────────────────────

/// The JSON body POSTed to the external evaluator endpoint.
#[derive(Debug, Serialize)]
struct CalloutRequest<'a> {
    messages: &'a Vec<switchboard_common::types::Message>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    team: Option<&'a str>,
    metadata: &'a HashMap<String, String>,
}

/// The JSON body returned by the external evaluator.
#[derive(Debug, Deserialize)]
struct CalloutResponse {
    action: String,
    #[serde(default)]
    rule: String,
    #[serde(default)]
    reason: String,
    #[serde(default = "default_confidence")]
    confidence: f64,
    #[serde(default)]
    modified_content: Option<String>,
}

fn default_confidence() -> f64 {
    1.0
}

// ── HttpCalloutEngine ─────────────────────────────────────────────────────────

/// Guardrail engine that evaluates content by calling an external HTTP service.
///
/// Both pre-request and post-response evaluation POST to the same configured
/// endpoint.  The external service is responsible for determining which phase
/// applies based on its own configuration.
#[derive(Debug)]
pub struct HttpCalloutEngine {
    client: reqwest::Client,
    endpoint: String,
    timeout: Duration,
    fail_mode: FailMode,
    extra_headers: HashMap<String, String>,
    engine_name: String,
}

impl HttpCalloutEngine {
    /// Construct a new engine with explicit parameters.
    pub fn new(
        endpoint: impl Into<String>,
        timeout: Duration,
        fail_mode: FailMode,
        extra_headers: HashMap<String, String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.into(),
            timeout,
            fail_mode,
            extra_headers,
            engine_name: "http_callout".into(),
        }
    }

    /// Construct an engine from an [`EngineConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`ServerError::Config`] if the `endpoint` field is missing.
    pub fn from_config(cfg: &EngineConfig) -> Result<Self, ServerError> {
        let endpoint = cfg
            .endpoint
            .clone()
            .ok_or_else(|| ServerError::Config("http engine requires 'endpoint' field".into()))?;

        let timeout = cfg
            .timeout
            .as_deref()
            .map(parse_duration_cfg)
            .unwrap_or(Duration::from_millis(500));

        // Default fail mode for callout engines is Open (permissive).
        let fail_mode = FailMode::Open;

        Ok(Self::new(endpoint, timeout, fail_mode, cfg.headers.clone()))
    }

    /// Core evaluation logic shared by [`evaluate_request`] and
    /// [`evaluate_response`].
    async fn call_endpoint(&self, input: &GuardrailInput) -> Result<GuardrailVerdict, ServerError> {
        let start = Instant::now();

        let body = CalloutRequest {
            messages: &input.messages,
            model: &input.model,
            user_id: input.user.as_ref().map(|u| u.id.as_str()),
            team: input.user.as_ref().and_then(|u| u.team.as_deref()),
            metadata: &input.metadata,
        };

        // Build request with all configured extra headers.
        let mut request_builder = self
            .client
            .post(&self.endpoint)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .json(&body);

        for (name, value) in &self.extra_headers {
            request_builder = request_builder.header(name.as_str(), value.as_str());
        }

        // Apply per-engine timeout.
        let response_result = tokio::time::timeout(self.timeout, request_builder.send()).await;

        let http_response = match response_result {
            Ok(Ok(resp)) => resp,
            Ok(Err(transport_err)) => {
                warn!(
                    endpoint = %self.endpoint,
                    error = %transport_err,
                    "http callout transport error"
                );
                return self.fail_verdict("transport_error");
            }
            Err(_elapsed) => {
                warn!(
                    endpoint = %self.endpoint,
                    timeout_ms = self.timeout.as_millis(),
                    "http callout timed out"
                );
                return self.fail_verdict("timeout");
            }
        };

        // Non-2xx response → apply fail mode.
        if !http_response.status().is_success() {
            warn!(
                endpoint = %self.endpoint,
                status = %http_response.status(),
                "http callout returned non-2xx status"
            );
            return self.fail_verdict("non_2xx");
        }

        // Parse the response body.
        let callout_resp: CalloutResponse = http_response
            .json()
            .await
            .map_err(|e| ServerError::Config(format!("http callout response parse error: {e}")))?;

        let latency = start.elapsed();

        let action = map_action(&callout_resp);

        debug!(
            engine = self.engine_name,
            action = %action,
            rule = %callout_resp.rule,
            confidence = callout_resp.confidence,
            "http callout verdict"
        );

        Ok(GuardrailVerdict {
            action,
            engine: self.engine_name.clone(),
            rule: callout_resp.rule,
            reason: if callout_resp.reason.is_empty() {
                None
            } else {
                Some(callout_resp.reason)
            },
            confidence: callout_resp.confidence,
            latency,
        })
    }

    /// Produce a fail-mode verdict (Pass or Block depending on [`FailMode`]).
    fn fail_verdict(&self, rule: &str) -> Result<GuardrailVerdict, ServerError> {
        match self.fail_mode {
            FailMode::Open => Ok(GuardrailVerdict::pass(&self.engine_name)),
            FailMode::Closed => Ok(GuardrailVerdict {
                action: GuardrailAction::Block {
                    message: format!(
                        "Guardrail engine '{}' unavailable: request blocked",
                        self.engine_name
                    ),
                },
                engine: self.engine_name.clone(),
                rule: rule.into(),
                reason: Some(format!(
                    "HTTP callout to '{}' failed ({})",
                    self.endpoint, rule
                )),
                confidence: 1.0,
                latency: Duration::ZERO,
            }),
        }
    }
}

// ── GuardrailEngine impl ──────────────────────────────────────────────────────

#[async_trait]
impl GuardrailEngine for HttpCalloutEngine {
    fn name(&self) -> &str {
        &self.engine_name
    }

    /// Evaluate a prompt before it is forwarded to the LLM.
    async fn evaluate_request(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, ServerError> {
        self.call_endpoint(input).await
    }

    /// Evaluate a completion after it is received from the LLM.
    async fn evaluate_response(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, ServerError> {
        self.call_endpoint(input).await
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Map the `action` string from the callout response to a [`GuardrailAction`].
fn map_action(resp: &CalloutResponse) -> GuardrailAction {
    match resp.action.trim().to_ascii_lowercase().as_str() {
        "pass" => GuardrailAction::Pass,
        "block" => GuardrailAction::Block {
            message: resp.reason.clone(),
        },
        "modify" => GuardrailAction::Modify {
            modified_content: resp.modified_content.clone().unwrap_or_default(),
        },
        "audit_log" => GuardrailAction::AuditLog {
            severity: AuditSeverity::Warning,
        },
        other => {
            warn!(
                action = other,
                "unknown http callout action; treating as pass"
            );
            GuardrailAction::Pass
        }
    }
}

/// Parse a duration string such as `"300ms"` or `"2s"` into a [`Duration`].
/// Falls back to 500 ms on any parse error.
fn parse_duration_cfg(s: &str) -> Duration {
    let s = s.trim();
    if let Some(ms_str) = s.strip_suffix("ms") {
        if let Ok(ms) = ms_str.trim().parse::<u64>() {
            return Duration::from_millis(ms);
        }
    }
    if let Some(s_str) = s.strip_suffix('s') {
        if let Ok(secs) = s_str.trim().parse::<u64>() {
            return Duration::from_secs(secs);
        }
    }
    Duration::from_millis(500)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use super::*;
    use crate::guardrails::pipeline::FailMode;

    fn make_engine(endpoint: &str, fail_mode: FailMode) -> HttpCalloutEngine {
        HttpCalloutEngine::new(
            endpoint.to_owned(),
            Duration::from_millis(500),
            fail_mode,
            HashMap::new(),
        )
    }

    // ── map_action tests ──────────────────────────────────────────────────────

    fn resp(action: &str, reason: &str, modified_content: Option<&str>) -> CalloutResponse {
        CalloutResponse {
            action: action.into(),
            rule: "test_rule".into(),
            reason: reason.into(),
            confidence: 0.9,
            modified_content: modified_content.map(str::to_owned),
        }
    }

    #[test]
    fn test_map_action_pass() {
        let r = resp("pass", "", None);
        assert!(map_action(&r).is_pass());
    }

    #[test]
    fn test_map_action_block() {
        let r = resp("block", "blocked because reasons", None);
        assert!(map_action(&r).is_blocking());
        if let GuardrailAction::Block { message } = map_action(&r) {
            assert_eq!(message, "blocked because reasons");
        }
    }

    #[test]
    fn test_map_action_modify() {
        let r = resp("modify", "", Some("cleaned content"));
        if let GuardrailAction::Modify { modified_content } = map_action(&r) {
            assert_eq!(modified_content, "cleaned content");
        } else {
            panic!("expected Modify action");
        }
    }

    #[test]
    fn test_map_action_audit_log() {
        let r = resp("audit_log", "", None);
        assert!(matches!(map_action(&r), GuardrailAction::AuditLog { .. }));
    }

    #[test]
    fn test_map_action_unknown_is_pass() {
        let r = resp("unknown_action", "", None);
        assert!(map_action(&r).is_pass());
    }

    #[test]
    fn test_map_action_case_insensitive() {
        let r = resp("PASS", "", None);
        assert!(map_action(&r).is_pass());
        let r2 = resp("Block", "x", None);
        assert!(map_action(&r2).is_blocking());
    }

    // ── parse_duration_cfg tests ──────────────────────────────────────────────

    #[test]
    fn test_parse_duration_ms() {
        assert_eq!(parse_duration_cfg("300ms"), Duration::from_millis(300));
    }

    #[test]
    fn test_parse_duration_s() {
        assert_eq!(parse_duration_cfg("2s"), Duration::from_secs(2));
    }

    #[test]
    fn test_parse_duration_fallback() {
        assert_eq!(parse_duration_cfg("invalid"), Duration::from_millis(500));
    }

    // ── from_config tests ─────────────────────────────────────────────────────

    #[test]
    fn test_from_config_missing_endpoint_errors() {
        use crate::config::guardrails::EngineConfig;
        let cfg = EngineConfig {
            engine_type: "http".into(),
            phase: "pre_request".into(),
            rules: vec![],
            action: None,
            keywords: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            endpoint: None,
            timeout: None,
            tls: false,
            headers: Default::default(),
        };
        let result = HttpCalloutEngine::from_config(&cfg);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ServerError::Config(_)));
    }

    #[test]
    fn test_from_config_with_endpoint() {
        use crate::config::guardrails::EngineConfig;
        let mut headers = HashMap::new();
        headers.insert("Authorization".into(), "Bearer token".into());
        let cfg = EngineConfig {
            engine_type: "http".into(),
            phase: "pre_request".into(),
            rules: vec![],
            action: None,
            keywords: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            endpoint: Some("https://guardrails.internal/evaluate".into()),
            timeout: Some("300ms".into()),
            tls: false,
            headers,
        };
        let engine = HttpCalloutEngine::from_config(&cfg).unwrap();
        assert_eq!(engine.name(), "http_callout");
        assert_eq!(engine.endpoint, "https://guardrails.internal/evaluate");
        assert_eq!(engine.timeout, Duration::from_millis(300));
        assert_eq!(engine.extra_headers["Authorization"], "Bearer token");
    }

    // ── fail_verdict tests ────────────────────────────────────────────────────

    #[test]
    fn test_fail_open_returns_pass() {
        let engine = make_engine("http://localhost:9999/eval", FailMode::Open);
        let verdict = engine.fail_verdict("test").unwrap();
        assert!(verdict.action.is_pass());
    }

    #[test]
    fn test_fail_closed_returns_block() {
        let engine = make_engine("http://localhost:9999/eval", FailMode::Closed);
        let verdict = engine.fail_verdict("test").unwrap();
        assert!(verdict.action.is_blocking());
    }

    // ── engine name test ──────────────────────────────────────────────────────

    #[test]
    fn test_engine_name() {
        let engine = make_engine("http://localhost:9999", FailMode::Open);
        assert_eq!(engine.name(), "http_callout");
    }
}
