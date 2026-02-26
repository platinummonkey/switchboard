//! Keyword-based guardrail engine.
//!
//! Blocks (or applies a configured action to) requests that contain any keyword
//! from a blocklist.  Matching is case-insensitive substring search.

use std::time::Instant;

use async_trait::async_trait;
use tracing::debug;

use crate::error::ServerError;
use crate::guardrails::engine::{
    GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};

// ── KeywordEngine ─────────────────────────────────────────────────────────────

/// Guardrail engine that performs case-insensitive keyword matching.
#[derive(Debug)]
pub struct KeywordEngine {
    /// Stored lower-cased for O(1) compare during matching.
    keywords: Vec<String>,
    action: GuardrailAction,
}

impl KeywordEngine {
    /// Create a new engine with the given keyword list and default action.
    ///
    /// Keywords are normalised to lower-case at construction time.
    pub fn new(keywords: Vec<String>, action: GuardrailAction) -> Self {
        let keywords = keywords
            .into_iter()
            .map(|k| k.to_ascii_lowercase())
            .collect();
        Self { keywords, action }
    }

    fn evaluate(&self, input: &GuardrailInput) -> GuardrailVerdict {
        let start = Instant::now();
        let lower = input.content.to_ascii_lowercase();
        for kw in &self.keywords {
            if lower.contains(kw.as_str()) {
                let latency = start.elapsed();
                debug!(engine = self.name(), keyword = %kw, "keyword matched");
                let action = match &self.action {
                    GuardrailAction::Pass => GuardrailAction::Pass,
                    GuardrailAction::Block { .. } => GuardrailAction::Block {
                        message: format!("Request blocked: keyword '{kw}' found"),
                    },
                    GuardrailAction::Modify { modified_content } => GuardrailAction::Modify {
                        modified_content: modified_content.clone(),
                    },
                    GuardrailAction::AuditLog { severity } => GuardrailAction::AuditLog {
                        severity: severity.clone(),
                    },
                };
                return GuardrailVerdict {
                    action,
                    engine: self.name().to_owned(),
                    rule: kw.clone(),
                    reason: Some(format!("Keyword '{kw}' found in content")),
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
impl GuardrailEngine for KeywordEngine {
    fn name(&self) -> &str {
        "keyword"
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

    fn block_action() -> GuardrailAction {
        GuardrailAction::Block {
            message: "blocked".into(),
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
    async fn test_keyword_engine_case_insensitive_block() {
        let engine = KeywordEngine::new(vec!["badword".into()], block_action());
        let input = make_input("This contains BADWORD in it");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.engine, "keyword");
    }

    #[tokio::test]
    async fn test_keyword_engine_passes_when_no_match() {
        let engine = KeywordEngine::new(vec!["badword".into()], block_action());
        let input = make_input("This is a perfectly clean sentence");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_keyword_engine_partial_word_matches() {
        // "bad" is a substring of "badminton" — substring match should fire
        let engine = KeywordEngine::new(vec!["bad".into()], block_action());
        let input = make_input("I love badminton");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(
            verdict.action.is_blocking(),
            "substring match should trigger"
        );
    }

    #[tokio::test]
    async fn test_keyword_engine_multiple_keywords_first_wins() {
        let engine = KeywordEngine::new(vec!["alpha".into(), "beta".into()], block_action());
        let input = make_input("beta test alpha run");
        // "alpha" appears first in the list, but "beta" appears earlier in text —
        // we check keywords in list order, so "alpha" fires first.
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.rule, "alpha");
    }

    #[tokio::test]
    async fn test_keyword_engine_evaluate_response() {
        let engine = KeywordEngine::new(vec!["secret".into()], block_action());
        let input = make_input("The response contains a secret value");
        let verdict = engine.evaluate_response(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
    }

    #[test]
    fn test_keyword_engine_lowercases_at_construction() {
        let engine = KeywordEngine::new(vec!["UPPER".into()], block_action());
        assert_eq!(engine.keywords, vec!["upper"]);
    }
}
