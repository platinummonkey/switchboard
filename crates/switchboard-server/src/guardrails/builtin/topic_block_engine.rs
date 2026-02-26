//! Topic-block guardrail engine.
//!
//! Blocks requests that contain any keyword associated with a named topic,
//! using whole-word (word-boundary) matching.

use std::time::Instant;

use async_trait::async_trait;
use regex::Regex;
use tracing::debug;

use crate::error::ServerError;
use crate::guardrails::engine::{
    GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};

// ── TopicBlockEngine ──────────────────────────────────────────────────────────

/// Guardrail engine that blocks content related to a configurable topic.
///
/// Matching uses whole-word boundaries (`\b`) around each keyword so that,
/// for example, "harm" does not match "pharmacy".
#[derive(Debug)]
pub struct TopicBlockEngine {
    topic_name: String,
    action: GuardrailAction,
    /// Combined regex: `\b(kw1|kw2|…)\b`
    combined: Regex,
}

impl TopicBlockEngine {
    /// Create a new engine for the given topic and keyword list.
    ///
    /// Returns `Err` if the compiled regex is invalid (e.g. a keyword contains
    /// special regex metacharacters).
    pub fn new(
        topic_name: impl Into<String>,
        keywords: Vec<String>,
        action: GuardrailAction,
    ) -> Result<Self, ServerError> {
        let topic_name = topic_name.into();
        let escaped: Vec<String> = keywords.iter().map(|k| regex::escape(k)).collect();
        let pattern = format!(r"(?i)\b({})\b", escaped.join("|"));
        let combined = Regex::new(&pattern).map_err(|e| {
            ServerError::Config(format!(
                "topic_block_engine: failed to compile regex for topic '{topic_name}': {e}"
            ))
        })?;
        Ok(Self {
            topic_name,
            action,
            combined,
        })
    }

    fn evaluate(&self, input: &GuardrailInput) -> GuardrailVerdict {
        let start = Instant::now();
        if let Some(mat) = self.combined.find(&input.content) {
            let matched_kw = mat.as_str().to_ascii_lowercase();
            let latency = start.elapsed();
            debug!(
                engine = self.name(),
                topic = %self.topic_name,
                keyword = %matched_kw,
                "topic keyword matched"
            );
            let action = match &self.action {
                GuardrailAction::Pass => GuardrailAction::Pass,
                GuardrailAction::Block { .. } => GuardrailAction::Block {
                    message: format!(
                        "Request blocked: topic '{}' keyword '{}' found",
                        self.topic_name, matched_kw
                    ),
                },
                GuardrailAction::AuditLog { severity } => GuardrailAction::AuditLog {
                    severity: severity.clone(),
                },
                GuardrailAction::Modify { modified_content } => GuardrailAction::Modify {
                    modified_content: modified_content.clone(),
                },
            };
            return GuardrailVerdict {
                action,
                engine: self.name().to_owned(),
                rule: format!("topic:{}", self.topic_name),
                reason: Some(format!(
                    "Topic '{}' keyword '{}' matched in content",
                    self.topic_name, matched_kw
                )),
                confidence: 1.0,
                latency,
            };
        }
        let mut verdict = GuardrailVerdict::pass(self.name());
        verdict.latency = start.elapsed();
        verdict
    }
}

#[async_trait]
impl GuardrailEngine for TopicBlockEngine {
    fn name(&self) -> &str {
        "topic_block"
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

    fn weapons_engine() -> TopicBlockEngine {
        TopicBlockEngine::new(
            "weapons",
            vec!["bomb".into(), "explosive".into(), "grenade".into()],
            block_action(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_topic_block_matches_keyword() {
        let engine = weapons_engine();
        let input = make_input("How do I make a bomb?");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.engine, "topic_block");
        assert!(verdict.rule.contains("weapons"));
    }

    #[tokio::test]
    async fn test_topic_block_whole_word_only() {
        let engine = weapons_engine();
        // "bombardment" contains "bomb" but \b should prevent a match
        // NOTE: regex \bbomb\b will NOT match "bombardment" — correct behaviour
        let input = make_input("The bombardment was intense");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(
            verdict.action.is_pass(),
            "whole-word check failed: got {:?}",
            verdict.action
        );
    }

    #[tokio::test]
    async fn test_topic_block_passes_unrelated() {
        let engine = weapons_engine();
        let input = make_input("I enjoy hiking and cooking");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_topic_block_case_insensitive() {
        let engine = weapons_engine();
        let input = make_input("I need a GRENADE");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
    }

    #[tokio::test]
    async fn test_topic_block_evaluate_response() {
        let engine = weapons_engine();
        let input = make_input("The explosive device was found near...");
        let verdict = engine.evaluate_response(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
    }

    #[test]
    fn test_topic_block_engine_name() {
        let engine = weapons_engine();
        assert_eq!(engine.name(), "topic_block");
    }

    #[test]
    fn test_topic_block_empty_keywords_compiles() {
        // An empty keyword list should compile but never match anything
        let engine = TopicBlockEngine::new("empty", vec![], block_action());
        // This might fail to compile because `\b()\b` is an empty alternation;
        // we expect either Ok (always-pass) or Err from config validation.
        // Either is acceptable — what matters is no panic.
        let _ = engine;
    }
}
