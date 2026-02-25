//! Guardrail engine trait and associated verdict/action types.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use switchboard_common::types::Message;

use crate::identity::UserIdentity;

// ── Audit severity ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditSeverity {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for AuditSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditSeverity::Info => write!(f, "info"),
            AuditSeverity::Warning => write!(f, "warning"),
            AuditSeverity::Critical => write!(f, "critical"),
        }
    }
}

// ── Guardrail action ──────────────────────────────────────────────────────────

/// What a guardrail engine decides to do with a request or response.
#[derive(Debug, Clone)]
pub enum GuardrailAction {
    /// Allow the content to pass through unchanged.
    Pass,
    /// Reject the request/response; return `message` to the client.
    Block { message: String },
    /// Rewrite the content before forwarding.
    Modify { modified_content: String },
    /// Allow through but emit an audit log entry.
    AuditLog { severity: AuditSeverity },
}

impl GuardrailAction {
    pub fn is_pass(&self) -> bool {
        matches!(self, GuardrailAction::Pass)
    }

    pub fn is_blocking(&self) -> bool {
        matches!(self, GuardrailAction::Block { .. })
    }
}

impl std::fmt::Display for GuardrailAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuardrailAction::Pass => write!(f, "pass"),
            GuardrailAction::Block { .. } => write!(f, "block"),
            GuardrailAction::Modify { .. } => write!(f, "modify"),
            GuardrailAction::AuditLog { severity } => write!(f, "audit_log:{severity}"),
        }
    }
}

// ── Guardrail verdict ─────────────────────────────────────────────────────────

/// The full result of one engine's evaluation of a request or response.
#[derive(Debug, Clone)]
pub struct GuardrailVerdict {
    /// The action the engine recommends.
    pub action: GuardrailAction,
    /// Name of the engine that produced this verdict.
    pub engine: String,
    /// Name of the rule that triggered (empty string if `Pass`).
    pub rule: String,
    /// Human-readable explanation of why this verdict was reached.
    pub reason: Option<String>,
    /// Engine's confidence in this verdict (0.0–1.0).
    pub confidence: f64,
    /// Wall-clock time the evaluation took.
    pub latency: Duration,
}

impl GuardrailVerdict {
    /// Convenience: a passing verdict produced in zero time.
    pub fn pass(engine: impl Into<String>) -> Self {
        Self {
            action: GuardrailAction::Pass,
            engine: engine.into(),
            rule: String::new(),
            reason: None,
            confidence: 1.0,
            latency: Duration::ZERO,
        }
    }
}

// ── Guardrail input ───────────────────────────────────────────────────────────

/// Everything an engine needs to evaluate a request or response.
#[derive(Debug, Clone)]
pub struct GuardrailInput {
    /// The text to evaluate (concatenated prompt or response).
    pub content: String,
    /// Full message context.
    pub messages: Vec<Message>,
    /// Resolved user identity, if available.
    pub user: Option<UserIdentity>,
    /// The model name being targeted.
    pub model: String,
    /// Arbitrary key-value metadata (e.g. session ID, tool name).
    pub metadata: HashMap<String, String>,
}

// ── GuardrailEngine trait ─────────────────────────────────────────────────────

/// Pluggable guardrail evaluation engine.
///
/// Built-in implementations: `RegexEngine`, `KeywordEngine`,
/// `TokenLimitEngine`, `SecretDetectionEngine` (Phase 10a).
/// External: `GrpcCalloutEngine`, `HttpCalloutEngine` (Phase 10b/c).
#[async_trait]
pub trait GuardrailEngine: Send + Sync {
    /// Short unique name used in spans and metrics.
    fn name(&self) -> &str;

    /// Evaluate a prompt **before** it is forwarded to the LLM.
    async fn evaluate_request(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, crate::error::ServerError>;

    /// Evaluate a completion **after** it is received from the LLM.
    async fn evaluate_response(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, crate::error::ServerError>;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `GuardrailEngine` is object-safe.
    fn _assert_object_safe(_: &dyn GuardrailEngine) {}

    /// Stub engine that always passes.
    struct PassEngine;

    #[async_trait]
    impl GuardrailEngine for PassEngine {
        fn name(&self) -> &str {
            "pass"
        }
        async fn evaluate_request(
            &self,
            _input: &GuardrailInput,
        ) -> Result<GuardrailVerdict, crate::error::ServerError> {
            Ok(GuardrailVerdict::pass(self.name()))
        }
        async fn evaluate_response(
            &self,
            _input: &GuardrailInput,
        ) -> Result<GuardrailVerdict, crate::error::ServerError> {
            Ok(GuardrailVerdict::pass(self.name()))
        }
    }

    /// Stub engine that always blocks.
    struct BlockEngine;

    #[async_trait]
    impl GuardrailEngine for BlockEngine {
        fn name(&self) -> &str {
            "block"
        }
        async fn evaluate_request(
            &self,
            _input: &GuardrailInput,
        ) -> Result<GuardrailVerdict, crate::error::ServerError> {
            Ok(GuardrailVerdict {
                action: GuardrailAction::Block {
                    message: "Blocked by policy".into(),
                },
                engine: self.name().into(),
                rule: "always_block".into(),
                reason: Some("test policy".into()),
                confidence: 1.0,
                latency: Duration::from_millis(1),
            })
        }
        async fn evaluate_response(
            &self,
            _input: &GuardrailInput,
        ) -> Result<GuardrailVerdict, crate::error::ServerError> {
            Ok(GuardrailVerdict::pass(self.name()))
        }
    }

    fn make_input(content: &str) -> GuardrailInput {
        GuardrailInput {
            content: content.into(),
            messages: vec![],
            user: None,
            model: "claude-sonnet-4-20250514".into(),
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_pass_engine() {
        let e = PassEngine;
        let input = make_input("hello");
        let verdict = e.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_pass());
        assert_eq!(verdict.engine, "pass");
    }

    #[tokio::test]
    async fn test_block_engine() {
        let e = BlockEngine;
        let input = make_input("do something bad");
        let verdict = e.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.rule, "always_block");
    }

    #[test]
    fn test_action_helpers() {
        assert!(GuardrailAction::Pass.is_pass());
        assert!(!GuardrailAction::Pass.is_blocking());
        assert!(
            GuardrailAction::Block {
                message: "x".into()
            }
            .is_blocking()
        );
        assert!(
            !GuardrailAction::Block {
                message: "x".into()
            }
            .is_pass()
        );
    }

    #[test]
    fn test_action_display() {
        assert_eq!(GuardrailAction::Pass.to_string(), "pass");
        assert_eq!(
            GuardrailAction::Block {
                message: "x".into()
            }
            .to_string(),
            "block"
        );
        assert_eq!(
            GuardrailAction::Modify {
                modified_content: "y".into()
            }
            .to_string(),
            "modify"
        );
        assert_eq!(
            GuardrailAction::AuditLog {
                severity: AuditSeverity::Warning
            }
            .to_string(),
            "audit_log:warning"
        );
    }

    #[test]
    fn test_verdict_pass_convenience() {
        let v = GuardrailVerdict::pass("my-engine");
        assert!(v.action.is_pass());
        assert_eq!(v.engine, "my-engine");
        assert_eq!(v.latency, Duration::ZERO);
        assert!((v.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audit_severity_display() {
        assert_eq!(AuditSeverity::Info.to_string(), "info");
        assert_eq!(AuditSeverity::Warning.to_string(), "warning");
        assert_eq!(AuditSeverity::Critical.to_string(), "critical");
    }
}
