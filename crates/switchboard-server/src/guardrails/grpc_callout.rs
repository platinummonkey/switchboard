//! gRPC external guardrail callout engine.
//!
//! Calls an external gRPC service that implements `guardrails.v1.GuardrailEvaluator`
//! to evaluate requests and completions. Supports configurable timeouts and
//! fail-open / fail-closed behaviour on transport errors.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use switchboard_common::types::MessageContent;
use tonic::transport::Channel;
use tracing::{debug, error, warn};

use crate::error::ServerError;
use crate::guardrails::engine::{
    AuditSeverity, GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};
use crate::guardrails::pipeline::FailMode;
use crate::guardrails::proto::guardrails_v1::{
    Action, EvaluateCompletionInput, EvaluateRequestInput, Message as ProtoMessage,
    guardrail_evaluator_client::GuardrailEvaluatorClient,
};

// ── GrpcCalloutEngine ─────────────────────────────────────────────────────────

/// A guardrail engine that delegates evaluation to an external gRPC service.
///
/// The remote service must implement `guardrails.v1.GuardrailEvaluator`.
pub struct GrpcCalloutEngine {
    client: GuardrailEvaluatorClient<Channel>,
    timeout: Duration,
    fail_mode: FailMode,
    name: String,
}

impl std::fmt::Debug for GrpcCalloutEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcCalloutEngine")
            .field("name", &self.name)
            .field("timeout", &self.timeout)
            .field("fail_mode", &self.fail_mode)
            .finish()
    }
}

impl GrpcCalloutEngine {
    /// Connect to the gRPC endpoint.
    ///
    /// `endpoint` should be a full URI: `"http://host:port"` (plain) or
    /// `"https://host:port"` (TLS).
    pub async fn connect(
        endpoint: impl Into<String>,
        timeout: Duration,
        fail_mode: FailMode,
    ) -> Result<Self, ServerError> {
        let endpoint_str = endpoint.into();
        debug!(endpoint = %endpoint_str, "connecting to gRPC guardrail evaluator");

        let channel = Channel::from_shared(endpoint_str.clone())
            .map_err(|e| {
                ServerError::Config(format!("invalid gRPC endpoint '{endpoint_str}': {e}"))
            })?
            .connect()
            .await?;

        let client = GuardrailEvaluatorClient::new(channel);

        Ok(Self {
            client,
            timeout,
            fail_mode,
            name: "grpc_callout".into(),
        })
    }

    /// Apply `fail_mode` semantics after a transport/timeout failure.
    fn fail_verdict(&self, engine_name: &str, reason: &str) -> GuardrailVerdict {
        match self.fail_mode {
            FailMode::Open => {
                warn!(engine = %engine_name, %reason, "gRPC callout failed; fail-open → pass");
                GuardrailVerdict::pass(engine_name)
            }
            FailMode::Closed => {
                error!(engine = %engine_name, %reason, "gRPC callout failed; fail-closed → block");
                GuardrailVerdict {
                    action: GuardrailAction::Block {
                        message: format!(
                            "Guardrail engine '{engine_name}' unavailable: request blocked"
                        ),
                    },
                    engine: engine_name.into(),
                    rule: "engine_error".into(),
                    reason: Some(reason.into()),
                    confidence: 1.0,
                    latency: Duration::ZERO,
                }
            }
        }
    }
}

// ── Proto helpers ─────────────────────────────────────────────────────────────

/// Convert a slice of internal `Message` values to proto `Message`s.
fn to_proto_messages(msgs: &[switchboard_common::types::Message]) -> Vec<ProtoMessage> {
    msgs.iter()
        .map(|m| {
            let role = format!("{:?}", m.role).to_ascii_lowercase();
            let content = match &m.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::Parts(_) => m.content.as_text(),
            };
            ProtoMessage { role, content }
        })
        .collect()
}

/// Map a proto `Action` integer (as returned by `EvaluateResponse`) to a
/// `GuardrailAction`.
fn map_action(action_i32: i32, reason: &str, modified_content: &str) -> GuardrailAction {
    match Action::try_from(action_i32).unwrap_or(Action::Pass) {
        Action::Pass => GuardrailAction::Pass,
        Action::Block => GuardrailAction::Block {
            message: reason.to_owned(),
        },
        Action::Modify => GuardrailAction::Modify {
            modified_content: modified_content.to_owned(),
        },
        Action::AuditLog => GuardrailAction::AuditLog {
            severity: AuditSeverity::Warning,
        },
    }
}

// ── GuardrailEngine implementation ───────────────────────────────────────────

#[async_trait]
impl GuardrailEngine for GrpcCalloutEngine {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate_request(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, ServerError> {
        let engine_name = self.name.clone();
        let start = Instant::now();

        let proto_request = EvaluateRequestInput {
            messages: to_proto_messages(&input.messages),
            model: input.model.clone(),
            user_id: input
                .user
                .as_ref()
                .map(|u| u.id.clone())
                .unwrap_or_default(),
            team: input
                .user
                .as_ref()
                .and_then(|u| u.team.clone())
                .unwrap_or_default(),
            metadata: input.metadata.clone(),
        };

        let mut client = self.client.clone();
        let call_future = client.evaluate_request(tonic::Request::new(proto_request));

        match tokio::time::timeout(self.timeout, call_future).await {
            Ok(Ok(response)) => {
                let resp = response.into_inner();
                let latency = start.elapsed();
                let action = map_action(resp.action, &resp.reason, &resp.modified_content);

                debug!(
                    engine = %engine_name,
                    rule = %resp.rule,
                    confidence = resp.confidence,
                    "gRPC evaluate_request returned"
                );

                Ok(GuardrailVerdict {
                    action,
                    engine: engine_name,
                    rule: resp.rule,
                    reason: if resp.reason.is_empty() {
                        None
                    } else {
                        Some(resp.reason)
                    },
                    confidence: resp.confidence,
                    latency,
                })
            }
            Ok(Err(status)) => {
                let reason = format!("gRPC status {}: {}", status.code(), status.message());
                error!(engine = %engine_name, %reason, "evaluate_request RPC failed");
                Ok(self.fail_verdict(&engine_name, &reason))
            }
            Err(_elapsed) => {
                let reason = format!(
                    "gRPC evaluate_request timed out after {}ms",
                    self.timeout.as_millis()
                );
                warn!(engine = %engine_name, %reason, "evaluate_request timed out");
                Ok(self.fail_verdict(&engine_name, &reason))
            }
        }
    }

    async fn evaluate_response(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, ServerError> {
        let engine_name = self.name.clone();
        let start = Instant::now();

        let proto_request = EvaluateCompletionInput {
            completion: input.content.clone(),
            messages: to_proto_messages(&input.messages),
            model: input.model.clone(),
            user_id: input
                .user
                .as_ref()
                .map(|u| u.id.clone())
                .unwrap_or_default(),
            metadata: input.metadata.clone(),
        };

        let mut client = self.client.clone();
        let call_future = client.evaluate_completion(tonic::Request::new(proto_request));

        match tokio::time::timeout(self.timeout, call_future).await {
            Ok(Ok(response)) => {
                let resp = response.into_inner();
                let latency = start.elapsed();
                let action = map_action(resp.action, &resp.reason, &resp.modified_content);

                debug!(
                    engine = %engine_name,
                    rule = %resp.rule,
                    confidence = resp.confidence,
                    "gRPC evaluate_completion returned"
                );

                Ok(GuardrailVerdict {
                    action,
                    engine: engine_name,
                    rule: resp.rule,
                    reason: if resp.reason.is_empty() {
                        None
                    } else {
                        Some(resp.reason)
                    },
                    confidence: resp.confidence,
                    latency,
                })
            }
            Ok(Err(status)) => {
                let reason = format!("gRPC status {}: {}", status.code(), status.message());
                error!(engine = %engine_name, %reason, "evaluate_completion RPC failed");
                Ok(self.fail_verdict(&engine_name, &reason))
            }
            Err(_elapsed) => {
                let reason = format!(
                    "gRPC evaluate_completion timed out after {}ms",
                    self.timeout.as_millis()
                );
                warn!(engine = %engine_name, %reason, "evaluate_completion timed out");
                Ok(self.fail_verdict(&engine_name, &reason))
            }
        }
    }
}

// ── async_engine_from_config factory ─────────────────────────────────────────

use crate::config::guardrails::EngineConfig;

/// Build engines that require async setup (gRPC, HTTP callout).
///
/// Falls back to [`crate::guardrails::builtin::engine_from_config`] for
/// sync-constructable built-in engine types.
pub async fn async_engine_from_config(
    cfg: &EngineConfig,
) -> Result<Box<dyn GuardrailEngine>, ServerError> {
    match cfg.engine_type.as_str() {
        "grpc" => {
            let raw_endpoint = cfg
                .endpoint
                .as_deref()
                .ok_or_else(|| ServerError::Config("grpc engine requires 'endpoint'".into()))?;

            // Parse the timeout string if provided, otherwise default to 500 ms.
            let timeout = cfg
                .timeout
                .as_deref()
                .map(parse_timeout)
                .unwrap_or(Duration::from_millis(500));

            let fail_mode = FailMode::Open; // future: cfg.fail_mode field

            // Normalise the endpoint: add scheme if missing.
            let endpoint =
                if raw_endpoint.starts_with("http://") || raw_endpoint.starts_with("https://") {
                    raw_endpoint.to_owned()
                } else if cfg.tls {
                    format!("https://{raw_endpoint}")
                } else {
                    format!("http://{raw_endpoint}")
                };

            let engine = GrpcCalloutEngine::connect(endpoint, timeout, fail_mode).await?;
            Ok(Box::new(engine))
        }
        _ => {
            // Delegate to the sync factory for the 5 built-in types.
            crate::guardrails::builtin::engine_from_config(cfg)
        }
    }
}

/// Parse a human-friendly duration string (e.g. `"200ms"`, `"1s"`).
/// Falls back to 500 ms on any parse error.
fn parse_timeout(s: &str) -> Duration {
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
