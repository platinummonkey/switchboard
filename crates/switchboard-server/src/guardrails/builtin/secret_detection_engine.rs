//! Secret-detection guardrail engine.
//!
//! Scans content for common secret patterns (API keys, private key PEM blocks,
//! password-in-URL, etc.) using pre-compiled regular expressions.

use std::time::Instant;

use async_trait::async_trait;
use regex::Regex;
use tracing::debug;

use crate::error::ServerError;
use crate::guardrails::engine::{
    AuditSeverity, GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};

// ── SecretDetectionEngine ─────────────────────────────────────────────────────

/// Guardrail engine that detects common secret patterns.
#[derive(Debug)]
pub struct SecretDetectionEngine {
    action: GuardrailAction,
    /// Compiled patterns stored as (name, Regex) pairs.
    patterns: Vec<(&'static str, Regex)>,
}

impl SecretDetectionEngine {
    /// Build the engine with the given default action.
    ///
    /// Patterns compiled at construction time:
    /// - Anthropic API key (`sk-ant-…`)
    /// - OpenAI API key (`sk-…`)
    /// - AWS access key (`AKIA…`)
    /// - Generic API key assignment
    /// - Private key PEM header
    /// - Password embedded in URL
    pub fn new(action: GuardrailAction) -> Self {
        // Use a raw string that avoids the `\"` issue; quote characters are
        // expressed via a character class instead.
        let generic_key_pattern = concat!(
            r"\b(api[_-]?key|apikey)\s*[:=]\s*[",
            r#"'"]?"#,
            r"[A-Za-z0-9\-_]{16,}"
        );

        let patterns: Vec<(&'static str, Regex)> = vec![
            (
                "anthropic_api_key",
                Regex::new(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")
                    .expect("anthropic key pattern is valid"),
            ),
            (
                "openai_api_key",
                Regex::new(r"\bsk-[A-Za-z0-9]{20,}\b").expect("openai key pattern is valid"),
            ),
            (
                "aws_access_key",
                Regex::new(r"\bAKIA[0-9A-Z]{16}\b").expect("aws key pattern is valid"),
            ),
            (
                "generic_api_key",
                Regex::new(generic_key_pattern).expect("generic api key pattern is valid"),
            ),
            (
                "private_key_pem",
                Regex::new(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----")
                    .expect("pem key pattern is valid"),
            ),
            (
                "password_in_url",
                Regex::new(r"://[^:@\s]+:[^@\s]+@").expect("password in url pattern is valid"),
            ),
        ];
        Self { action, patterns }
    }

    fn evaluate(&self, input: &GuardrailInput) -> GuardrailVerdict {
        let start = Instant::now();
        for (name, pattern) in &self.patterns {
            if pattern.is_match(&input.content) {
                let latency = start.elapsed();
                debug!(
                    engine = self.name(),
                    pattern = name,
                    "secret pattern matched"
                );
                let action = match &self.action {
                    GuardrailAction::Pass => GuardrailAction::Pass,
                    GuardrailAction::Block { .. } => GuardrailAction::Block {
                        message: format!("Secret detected: pattern '{name}' matched"),
                    },
                    GuardrailAction::AuditLog { .. } => GuardrailAction::AuditLog {
                        severity: AuditSeverity::Critical,
                    },
                    GuardrailAction::Modify { modified_content } => GuardrailAction::Modify {
                        modified_content: modified_content.clone(),
                    },
                };
                return GuardrailVerdict {
                    action,
                    engine: self.name().to_owned(),
                    rule: name.to_string(),
                    reason: Some(format!("Secret pattern '{name}' matched in content")),
                    confidence: 1.0,
                    latency,
                };
            }
        }
        let mut verdict = GuardrailVerdict::pass(self.name());
        verdict.latency = start.elapsed();
        verdict
    }
}

#[async_trait]
impl GuardrailEngine for SecretDetectionEngine {
    fn name(&self) -> &str {
        "secret_detection"
    }

    async fn evaluate_request(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, ServerError> {
        Ok(self.evaluate(input))
    }

    async fn evaluate_response(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, ServerError> {
        Ok(self.evaluate(input))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn block_engine() -> SecretDetectionEngine {
        SecretDetectionEngine::new(GuardrailAction::Block {
            message: "secret detected".into(),
        })
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
    async fn test_secret_detection_finds_anthropic_key() {
        let engine = block_engine();
        let input = make_input("Here is my key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456789");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(
            verdict.action.is_blocking(),
            "expected block, got {:?}",
            verdict.action
        );
        assert_eq!(verdict.rule, "anthropic_api_key");
    }

    #[tokio::test]
    async fn test_secret_detection_finds_openai_key() {
        let engine = block_engine();
        // Classic OpenAI key format: sk- followed by 48 alphanumeric characters
        let input = make_input("token=sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN12345678");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        // The openai pattern fires (anthropic check is for sk-ant- prefix which is absent)
        assert!(
            verdict.rule == "openai_api_key",
            "unexpected rule: {}",
            verdict.rule
        );
    }

    #[tokio::test]
    async fn test_secret_detection_finds_aws_key() {
        let engine = block_engine();
        let input = make_input("aws_access_key_id = AKIAIOSFODNN7EXAMPLE");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.rule, "aws_access_key");
    }

    #[tokio::test]
    async fn test_secret_detection_finds_pem_key() {
        let engine = block_engine();
        let input = make_input("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAK...");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.rule, "private_key_pem");
    }

    #[tokio::test]
    async fn test_secret_detection_finds_ec_pem_key() {
        let engine = block_engine();
        let input = make_input("-----BEGIN EC PRIVATE KEY-----\nMHQCAQEEI...");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.rule, "private_key_pem");
    }

    #[tokio::test]
    async fn test_secret_detection_finds_generic_api_key() {
        let engine = block_engine();
        let input = make_input("api_key=abcdef1234567890abcdef");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.rule, "generic_api_key");
    }

    #[tokio::test]
    async fn test_secret_detection_finds_password_in_url() {
        let engine = block_engine();
        let input = make_input("Connect via postgresql://user:s3cret@db.example.com/mydb");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.rule, "password_in_url");
    }

    #[tokio::test]
    async fn test_secret_detection_clean_content_passes() {
        let engine = block_engine();
        let input = make_input("What is the capital of France?");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_secret_detection_evaluate_response() {
        let engine = block_engine();
        let input = make_input("The secret is sk-ant-api03-zzzzzzzzzzzzzzzzzzzzzzzzzz");
        let verdict = engine.evaluate_response(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
    }
}
