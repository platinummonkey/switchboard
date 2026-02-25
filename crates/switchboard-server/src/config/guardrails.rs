//! Guardrail pipeline configuration.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

fn default_fail_mode() -> String {
    "open".into()
}

fn default_timeout() -> String {
    "500ms".into()
}

fn default_streaming_mode() -> String {
    "async_audit".into()
}

/// A regex-based guardrail rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegexRule {
    pub name: String,
    pub pattern: String,
    pub action: String,
}

/// Configuration for a single guardrail engine entry.
/// The `type` field selects the engine; all other fields are engine-specific.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Engine type: "builtin_regex" | "builtin_secret_detection" |
    /// "builtin_keyword" | "builtin_token_limit" | "grpc" | "http".
    #[serde(rename = "type")]
    pub engine_type: String,

    /// Evaluation phase: "pre_request" | "post_response".
    pub phase: String,

    // ── builtin_regex ─────────────────────────────────────────────────────────
    #[serde(default)]
    pub rules: Vec<RegexRule>,

    // ── builtin_secret_detection / builtin_keyword / builtin_token_limit ─────
    /// Default action when the engine triggers: "block" | "modify" | "audit_log".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,

    /// Keywords to block (builtin_keyword).
    #[serde(default)]
    pub keywords: Vec<String>,

    /// Max input tokens (builtin_token_limit).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u32>,

    /// Max output tokens (builtin_token_limit).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    // ── grpc / http callout ───────────────────────────────────────────────────
    /// Remote endpoint (grpc: "host:port", http: full URL).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,

    /// Per-callout timeout, overrides the global guardrail timeout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<String>,

    /// Enable TLS for gRPC callout.
    #[serde(default)]
    pub tls: bool,

    /// Extra headers to include in HTTP callout requests.
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

/// Top-level guardrail configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailsConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Behaviour when a guardrail engine errors: "open" (allow) | "closed" (block).
    #[serde(default = "default_fail_mode")]
    pub fail_mode: String,

    /// Maximum wall-clock time for all guardrail evaluation per request.
    #[serde(default = "default_timeout")]
    pub timeout: String,

    /// How post-response guardrails handle streaming: "buffered" | "chunked" | "async_audit".
    #[serde(default = "default_streaming_mode")]
    pub streaming_mode: String,

    #[serde(default)]
    pub engines: Vec<EngineConfig>,
}

impl Default for GuardrailsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fail_mode: default_fail_mode(),
            timeout: default_timeout(),
            streaming_mode: default_streaming_mode(),
            engines: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_builtin_regex() {
        let toml = r#"
enabled = true
fail_mode = "open"
timeout = "500ms"
streaming_mode = "async_audit"

[[engines]]
type = "builtin_regex"
phase = "pre_request"
rules = [
    { name = "ssn", pattern = "\\b\\d{3}-\\d{2}-\\d{4}\\b", action = "block" },
    { name = "api_key", pattern = "\\b(sk-|AKIA)[A-Za-z0-9]{20,}\\b", action = "block" },
]
"#;
        let c: GuardrailsConfig = toml::from_str(toml).unwrap();
        assert!(c.enabled);
        assert_eq!(c.fail_mode, "open");
        assert_eq!(c.engines.len(), 1);
        assert_eq!(c.engines[0].engine_type, "builtin_regex");
        assert_eq!(c.engines[0].phase, "pre_request");
        assert_eq!(c.engines[0].rules.len(), 2);
        assert_eq!(c.engines[0].rules[0].name, "ssn");
    }

    #[test]
    fn test_parse_grpc_callout() {
        let toml = r#"
enabled = true
fail_mode = "closed"
timeout = "1s"
streaming_mode = "buffered"

[[engines]]
type = "grpc"
phase = "pre_request"
endpoint = "guardrails.internal:50051"
tls = true
timeout = "200ms"
"#;
        let c: GuardrailsConfig = toml::from_str(toml).unwrap();
        let e = &c.engines[0];
        assert_eq!(e.engine_type, "grpc");
        assert_eq!(e.endpoint.as_deref(), Some("guardrails.internal:50051"));
        assert!(e.tls);
        assert_eq!(e.timeout.as_deref(), Some("200ms"));
    }

    #[test]
    fn test_parse_http_callout_with_headers() {
        let toml = r#"
enabled = true
fail_mode = "open"
timeout = "500ms"
streaming_mode = "async_audit"

[[engines]]
type = "http"
phase = "post_response"
endpoint = "https://guardrails.internal/evaluate"
timeout = "300ms"
[engines.headers]
Authorization = "Bearer secret"
"#;
        let c: GuardrailsConfig = toml::from_str(toml).unwrap();
        let e = &c.engines[0];
        assert_eq!(e.engine_type, "http");
        assert_eq!(e.headers["Authorization"], "Bearer secret");
    }

    #[test]
    fn test_defaults() {
        let c = GuardrailsConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.fail_mode, "open");
        assert_eq!(c.streaming_mode, "async_audit");
        assert!(c.engines.is_empty());
    }
}
