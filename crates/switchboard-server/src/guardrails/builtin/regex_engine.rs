//! Regex-based guardrail engine.
//!
//! Evaluates prompts against a list of compiled regular-expression rules.
//! Each rule carries its own name and recommended action.  The first matching
//! rule wins; if no rule matches the verdict is `Pass`.

use std::time::Instant;

use async_trait::async_trait;
use regex::Regex;
use tracing::debug;

use crate::config::guardrails::RegexRule;
use crate::error::ServerError;
use crate::guardrails::engine::{
    AuditSeverity, GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};

// ── Compiled rule ─────────────────────────────────────────────────────────────

struct CompiledRule {
    name: String,
    pattern: Regex,
    action: GuardrailAction,
}

// ── RegexEngine ───────────────────────────────────────────────────────────────

/// Guardrail engine that tests content against a list of regex rules.
#[derive(Debug)]
pub struct RegexEngine {
    rules: Vec<CompiledRule>,
}

impl std::fmt::Debug for CompiledRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledRule")
            .field("name", &self.name)
            .field("pattern", &self.pattern.as_str())
            .finish()
    }
}

/// Parse an action string into a [`GuardrailAction`].
///
/// Recognised strings: `"block"` (default), `"audit_log"`, `"pass"`.
fn action_from_rule_str(action: &str, rule_name: &str) -> GuardrailAction {
    match action.trim().to_ascii_lowercase().as_str() {
        "audit_log" => GuardrailAction::AuditLog {
            severity: AuditSeverity::Warning,
        },
        "pass" => GuardrailAction::Pass,
        _ => GuardrailAction::Block {
            message: format!("Request blocked by rule '{rule_name}'"),
        },
    }
}

impl RegexEngine {
    /// Build a [`RegexEngine`] from a slice of [`RegexRule`] config entries.
    ///
    /// Returns `Err` immediately if any pattern fails to compile.
    pub fn from_config(rules: &[RegexRule]) -> Result<Self, ServerError> {
        let compiled = rules
            .iter()
            .map(|r| {
                let pattern = Regex::new(&r.pattern).map_err(|e| {
                    ServerError::Config(format!(
                        "regex_engine: invalid pattern '{}' in rule '{}': {e}",
                        r.pattern, r.name
                    ))
                })?;
                let action = action_from_rule_str(&r.action, &r.name);
                Ok(CompiledRule {
                    name: r.name.clone(),
                    pattern,
                    action,
                })
            })
            .collect::<Result<Vec<_>, ServerError>>()?;

        Ok(Self { rules: compiled })
    }

    fn evaluate(&self, input: &GuardrailInput) -> GuardrailVerdict {
        let start = Instant::now();
        for rule in &self.rules {
            if rule.pattern.is_match(&input.content) {
                let latency = start.elapsed();
                debug!(
                    engine = self.name(),
                    rule = %rule.name,
                    "regex rule matched"
                );
                let action = match &rule.action {
                    GuardrailAction::Block { .. } => GuardrailAction::Block {
                        message: format!("Request blocked by rule '{}'", rule.name),
                    },
                    GuardrailAction::AuditLog { .. } => GuardrailAction::AuditLog {
                        severity: AuditSeverity::Warning,
                    },
                    other => other.clone(),
                };
                return GuardrailVerdict {
                    action,
                    engine: self.name().to_owned(),
                    rule: rule.name.clone(),
                    reason: Some(format!(
                        "Pattern '{}' matched content",
                        rule.pattern.as_str()
                    )),
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
impl GuardrailEngine for RegexEngine {
    fn name(&self) -> &str {
        "regex"
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

    fn make_input(content: &str) -> GuardrailInput {
        GuardrailInput {
            content: content.into(),
            messages: vec![],
            user: None,
            model: "claude-sonnet-4-20250514".into(),
            metadata: HashMap::new(),
        }
    }

    fn ssn_rule() -> RegexRule {
        RegexRule {
            name: "ssn".into(),
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".into(),
            action: "block".into(),
        }
    }

    fn audit_rule() -> RegexRule {
        RegexRule {
            name: "phone".into(),
            pattern: r"\b\d{3}-\d{3}-\d{4}\b".into(),
            action: "audit_log".into(),
        }
    }

    #[tokio::test]
    async fn test_regex_engine_blocks_matching_content() {
        let engine = RegexEngine::from_config(&[ssn_rule()]).unwrap();
        let input = make_input("My SSN is 123-45-6789");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(
            verdict.action.is_blocking(),
            "expected Block, got {:?}",
            verdict.action
        );
        assert_eq!(verdict.engine, "regex");
        assert_eq!(verdict.rule, "ssn");
    }

    #[tokio::test]
    async fn test_regex_engine_passes_clean_content() {
        let engine = RegexEngine::from_config(&[ssn_rule()]).unwrap();
        let input = make_input("What is the capital of France?");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_regex_engine_audit_log_action() {
        let engine = RegexEngine::from_config(&[audit_rule()]).unwrap();
        let input = make_input("Call me at 555-867-5309");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        assert!(
            matches!(verdict.action, GuardrailAction::AuditLog { .. }),
            "expected AuditLog, got {:?}",
            verdict.action
        );
        assert_eq!(verdict.rule, "phone");
    }

    #[test]
    fn test_regex_engine_invalid_pattern_returns_error() {
        let bad_rule = RegexRule {
            name: "bad".into(),
            pattern: r"[invalid(".into(),
            action: "block".into(),
        };
        let result = RegexEngine::from_config(&[bad_rule]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ServerError::Config(_)),
            "expected Config error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_regex_engine_first_match_wins() {
        let rules = vec![
            ssn_rule(),
            RegexRule {
                name: "digit_run".into(),
                pattern: r"\d{3}".into(),
                action: "audit_log".into(),
            },
        ];
        let engine = RegexEngine::from_config(&rules).unwrap();
        let input = make_input("SSN: 123-45-6789");
        let verdict = engine.evaluate_request(&input).await.unwrap();
        // First rule (ssn) should fire, not digit_run
        assert_eq!(verdict.rule, "ssn");
        assert!(verdict.action.is_blocking());
    }

    #[tokio::test]
    async fn test_regex_engine_evaluate_response() {
        let engine = RegexEngine::from_config(&[ssn_rule()]).unwrap();
        let input = make_input("The SSN was 987-65-4321 in the response");
        let verdict = engine.evaluate_response(&input).await.unwrap();
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.rule, "ssn");
    }
}
