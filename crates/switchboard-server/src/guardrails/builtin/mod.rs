//! Built-in guardrail engine implementations.
//!
//! This module re-exports all five built-in engines and provides
//! [`engine_from_config`] to instantiate an engine from an [`EngineConfig`].

pub mod keyword_engine;
pub mod regex_engine;
pub mod secret_detection_engine;
pub mod token_limit_engine;
pub mod topic_block_engine;

pub use keyword_engine::KeywordEngine;
pub use regex_engine::RegexEngine;
pub use secret_detection_engine::SecretDetectionEngine;
pub use token_limit_engine::TokenLimitEngine;
pub use topic_block_engine::TopicBlockEngine;

use crate::config::guardrails::EngineConfig;
use crate::error::ServerError;
use crate::guardrails::engine::{AuditSeverity, GuardrailAction, GuardrailEngine};

// ── Action helper ─────────────────────────────────────────────────────────────

/// Convert an optional action string from config into a [`GuardrailAction`].
///
/// Defaults to `Block` when `None` or unrecognised.
pub fn action_from_str(action: Option<&str>) -> GuardrailAction {
    match action
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("audit_log") => GuardrailAction::AuditLog {
            severity: AuditSeverity::Warning,
        },
        Some("pass") => GuardrailAction::Pass,
        _ => GuardrailAction::Block {
            message: "Request blocked by guardrail".into(),
        },
    }
}

// ── Factory function ──────────────────────────────────────────────────────────

/// Build a boxed [`GuardrailEngine`] from an [`EngineConfig`].
///
/// # Errors
///
/// Returns [`ServerError::Config`] when:
/// * the `engine_type` is not recognised, or
/// * a built-in engine fails to initialise (e.g. invalid regex pattern).
pub fn engine_from_config(cfg: &EngineConfig) -> Result<Box<dyn GuardrailEngine>, ServerError> {
    match cfg.engine_type.as_str() {
        "builtin_regex" => {
            let engine = RegexEngine::from_config(&cfg.rules)?;
            Ok(Box::new(engine))
        }
        "builtin_keyword" => {
            let action = action_from_str(cfg.action.as_deref());
            let engine = KeywordEngine::new(cfg.keywords.clone(), action);
            Ok(Box::new(engine))
        }
        "builtin_token_limit" => {
            let engine = TokenLimitEngine::new(cfg.max_input_tokens, cfg.max_output_tokens);
            Ok(Box::new(engine))
        }
        "builtin_secret_detection" => {
            let action = action_from_str(cfg.action.as_deref());
            let engine = SecretDetectionEngine::new(action);
            Ok(Box::new(engine))
        }
        "builtin_topic_block" => {
            let action = action_from_str(cfg.action.as_deref());
            // Use keywords from config; topic name comes from the first available
            // hint or defaults to "unnamed_topic".
            let topic_name = cfg
                .keywords
                .first()
                .map(|_| "configured_topic")
                .unwrap_or("unnamed_topic");
            let engine = TopicBlockEngine::new(topic_name, cfg.keywords.clone(), action)?;
            Ok(Box::new(engine))
        }
        other => Err(ServerError::Config(format!(
            "unknown engine type '{other}'"
        ))),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::guardrails::{EngineConfig, RegexRule};

    fn base_config(engine_type: &str) -> EngineConfig {
        EngineConfig {
            engine_type: engine_type.into(),
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
        }
    }

    #[test]
    fn test_engine_from_config_regex() {
        let mut cfg = base_config("builtin_regex");
        cfg.rules = vec![RegexRule {
            name: "ssn".into(),
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".into(),
            action: "block".into(),
        }];
        let engine = engine_from_config(&cfg).unwrap();
        assert_eq!(engine.name(), "regex");
    }

    #[test]
    fn test_engine_from_config_keyword() {
        let mut cfg = base_config("builtin_keyword");
        cfg.keywords = vec!["secret".into()];
        let engine = engine_from_config(&cfg).unwrap();
        assert_eq!(engine.name(), "keyword");
    }

    #[test]
    fn test_engine_from_config_token_limit() {
        let mut cfg = base_config("builtin_token_limit");
        cfg.max_input_tokens = Some(1000);
        let engine = engine_from_config(&cfg).unwrap();
        assert_eq!(engine.name(), "token_limit");
    }

    #[test]
    fn test_engine_from_config_secret_detection() {
        let cfg = base_config("builtin_secret_detection");
        let engine = engine_from_config(&cfg).unwrap();
        assert_eq!(engine.name(), "secret_detection");
    }

    #[test]
    fn test_engine_from_config_topic_block() {
        let mut cfg = base_config("builtin_topic_block");
        cfg.keywords = vec!["bomb".into(), "explosive".into()];
        let engine = engine_from_config(&cfg).unwrap();
        assert_eq!(engine.name(), "topic_block");
    }

    #[test]
    fn test_engine_from_config_unknown_type_errors() {
        let cfg = base_config("grpc");
        let result = engine_from_config(&cfg);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            matches!(err, ServerError::Config(_)),
            "expected Config error, got {err:?}"
        );
    }

    #[test]
    fn test_action_from_str_defaults_to_block() {
        let a = action_from_str(None);
        assert!(a.is_blocking());
    }

    #[test]
    fn test_action_from_str_audit_log() {
        let a = action_from_str(Some("audit_log"));
        assert!(matches!(a, GuardrailAction::AuditLog { .. }));
    }

    #[test]
    fn test_action_from_str_pass() {
        let a = action_from_str(Some("pass"));
        assert!(a.is_pass());
    }

    #[test]
    fn test_action_from_str_block_explicit() {
        let a = action_from_str(Some("block"));
        assert!(a.is_blocking());
    }
}
