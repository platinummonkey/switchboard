//! Token-limit guardrail engine.
//!
//! Enforces maximum token counts using a simple heuristic: 1 token ≈ 4 chars.
//! Pre-request evaluation checks `max_input_tokens`; post-response evaluation
//! checks `max_output_tokens`.

use std::time::Instant;

use async_trait::async_trait;
use tracing::debug;

use crate::error::ServerError;
use crate::guardrails::engine::{
    GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};

// ── TokenLimitEngine ──────────────────────────────────────────────────────────

/// Guardrail engine that rejects content exceeding configurable token limits.
#[derive(Debug)]
pub struct TokenLimitEngine {
    max_input_tokens: Option<u32>,
    max_output_tokens: Option<u32>,
}

impl TokenLimitEngine {
    /// Create a new engine.
    ///
    /// Either limit may be `None` to disable that direction's check.
    pub fn new(max_input: Option<u32>, max_output: Option<u32>) -> Self {
        Self {
            max_input_tokens: max_input,
            max_output_tokens: max_output,
        }
    }

    /// Estimate token count from character count (1 token ≈ 4 chars, minimum 1).
    fn estimate_tokens(text: &str) -> u32 {
        (text.len() as u32 / 4).max(1)
    }

    fn check_limit(&self, input: &GuardrailInput, limit: u32, direction: &str) -> GuardrailVerdict {
        let start = Instant::now();
        let estimated = Self::estimate_tokens(&input.content);
        let latency = start.elapsed();

        if estimated > limit {
            debug!(
                engine = self.name(),
                direction, estimated, limit, "token limit exceeded"
            );
            GuardrailVerdict {
                action: GuardrailAction::Block {
                    message: format!(
                        "Content exceeds {direction} token limit: estimated {estimated} > {limit}"
                    ),
                },
                engine: self.name().to_owned(),
                rule: format!("max_{direction}_tokens"),
                reason: Some(format!(
                    "Estimated {estimated} tokens exceeds limit of {limit}"
                )),
                confidence: 0.9,
                latency,
            }
        } else {
            let mut verdict = GuardrailVerdict::pass(self.name());
            verdict.latency = latency;
            verdict
        }
    }
}

#[async_trait]
impl GuardrailEngine for TokenLimitEngine {
    fn name(&self) -> &str {
        "token_limit"
    }

    async fn evaluate_request(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, ServerError> {
        if let Some(limit) = self.max_input_tokens {
            Ok(self.check_limit(input, limit, "input"))
        } else {
            Ok(GuardrailVerdict::pass(self.name()))
        }
    }

    async fn evaluate_response(
        &self,
        input: &GuardrailInput,
    ) -> Result<GuardrailVerdict, ServerError> {
        if let Some(limit) = self.max_output_tokens {
            Ok(self.check_limit(input, limit, "output"))
        } else {
            Ok(GuardrailVerdict::pass(self.name()))
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn make_input(content: &str) -> GuardrailInput {
        GuardrailInput {
            content: content.into(),
            messages: vec![],
            user: None,
            model: "claude-sonnet-4-20250514".into(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_estimate_tokens_basic() {
        assert_eq!(TokenLimitEngine::estimate_tokens(""), 1); // minimum 1
        assert_eq!(TokenLimitEngine::estimate_tokens("abcd"), 1); // 4 chars → 1 token
        assert_eq!(TokenLimitEngine::estimate_tokens("abcdefgh"), 2); // 8 chars → 2 tokens
        assert_eq!(TokenLimitEngine::estimate_tokens(&"x".repeat(400)), 100);
    }

    #[tokio::test]
    async fn test_token_limit_blocks_long_request() {
        // 100 chars → 25 tokens; set limit to 10
        let engine = TokenLimitEngine::new(Some(10), None);
        let input = make_input(&"a".repeat(100));
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.engine, "token_limit");
        assert!(verdict.rule.contains("input"));
    }

    #[tokio::test]
    async fn test_token_limit_passes_short_request() {
        let engine = TokenLimitEngine::new(Some(1000), None);
        let input = make_input("Short question");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_token_limit_response_check() {
        // 200 chars → 50 tokens; set output limit to 20
        let engine = TokenLimitEngine::new(None, Some(20));
        let input = make_input(&"b".repeat(200));
        // evaluate_request should pass (no input limit)
        let req_verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(req_verdict.action.is_pass());
        // evaluate_response should block (output limit exceeded)
        let resp_verdict = engine.evaluate_response(&input).await.unwrap();
        assert!(resp_verdict.action.is_blocking());
        assert!(resp_verdict.rule.contains("output"));
    }

    #[tokio::test]
    async fn test_token_limit_no_limits_always_pass() {
        let engine = TokenLimitEngine::new(None, None);
        let big_input = make_input(&"x".repeat(100_000));
        assert!(
            engine
                .evaluate_request(&big_input)
                .await
                .unwrap()
                .action
                .is_pass()
        );
        assert!(
            engine
                .evaluate_response(&big_input)
                .await
                .unwrap()
                .action
                .is_pass()
        );
    }

    #[tokio::test]
    async fn test_token_limit_exact_boundary_passes() {
        // 40 chars → 10 tokens; limit is exactly 10 → should pass (not exceed)
        let engine = TokenLimitEngine::new(Some(10), None);
        let input = make_input(&"a".repeat(40));
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_token_limit_one_over_boundary_blocks() {
        // 44 chars → 11 tokens; limit is 10 → should block
        let engine = TokenLimitEngine::new(Some(10), None);
        let input = make_input(&"a".repeat(44));
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
    }
}
