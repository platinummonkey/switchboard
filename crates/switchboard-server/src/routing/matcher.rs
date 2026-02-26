//! Rule matching for semantic routing.
//!
//! [`match_routing_rules`] evaluates a [`ClassificationResult`] against the
//! ordered list of [`RoutingRule`]s in a [`SemanticRoutingConfig`] and
//! returns the preferred models for the first matching rule.  If no rule
//! matches the default preferred models are returned.

use crate::config::routing::SemanticRoutingConfig;
use crate::routing::ClassificationResult;

/// Match a [`ClassificationResult`] against the rules in `config`.
///
/// # Matching semantics
///
/// A rule matches when **both** of the following hold:
///
/// 1. `rule.task_type == result.task_type.as_str()`
/// 2. `rule.complexity` is `None` (wildcard) **or**
///    `rule.complexity.as_deref() == Some(result.complexity.as_str())`
///
/// The first matching rule wins.  If no rule matches, `config.default.preferred_models`
/// is returned.
pub fn match_routing_rules(
    result: &ClassificationResult,
    config: &SemanticRoutingConfig,
) -> Vec<String> {
    let task_str = result.task_type.as_str();
    let complexity_str = result.complexity.as_str();

    for rule in &config.rules {
        if rule.task_type != task_str {
            continue;
        }
        if let Some(complexity_filter) = &rule.complexity {
            if complexity_filter.as_str() != complexity_str {
                continue;
            }
        }
        tracing::debug!(
            task_type = task_str,
            complexity = complexity_str,
            rule_task_type = rule.task_type.as_str(),
            "semantic routing: rule matched"
        );
        return rule.preferred_models.clone();
    }

    tracing::debug!(
        task_type = task_str,
        complexity = complexity_str,
        "semantic routing: no rule matched, using default"
    );
    config.default.preferred_models.clone()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::routing::{DefaultRouting, RoutingRule, SemanticRoutingConfig};
    use crate::routing::{ClassificationResult, Complexity, ModelRecommendation, TaskType};

    fn make_result(task: TaskType, complexity: Complexity) -> ClassificationResult {
        ClassificationResult {
            task_type: task,
            complexity,
            recommended_models: vec![ModelRecommendation {
                model: "claude-sonnet-4-20250514".into(),
                confidence: 0.8,
            }],
        }
    }

    fn make_config(rules: Vec<RoutingRule>, default: Vec<String>) -> SemanticRoutingConfig {
        SemanticRoutingConfig {
            enabled: true,
            classifier: "heuristic".into(),
            rules,
            default: DefaultRouting {
                preferred_models: default,
            },
        }
    }

    #[test]
    fn test_rule_matches_task_and_complexity() {
        let rules = vec![RoutingRule {
            task_type: "code_generation".into(),
            complexity: Some("complex".into()),
            preferred_models: vec!["claude-opus-4-20250514".into()],
            fallback_models: vec![],
        }];
        let config = make_config(rules, vec!["claude-sonnet-4-20250514".into()]);
        let result = make_result(
            TaskType::CodeGeneration { language: None },
            Complexity::Complex,
        );
        let models = match_routing_rules(&result, &config);
        assert_eq!(models, vec!["claude-opus-4-20250514"]);
    }

    #[test]
    fn test_rule_matches_task_any_complexity() {
        let rules = vec![RoutingRule {
            task_type: "tool_use".into(),
            complexity: None, // matches any complexity
            preferred_models: vec!["claude-sonnet-4-20250514".into()],
            fallback_models: vec![],
        }];
        let config = make_config(rules, vec!["claude-haiku-4-5-20251001".into()]);

        for complexity in [Complexity::Simple, Complexity::Medium, Complexity::Complex] {
            let result = make_result(TaskType::ToolUse, complexity);
            let models = match_routing_rules(&result, &config);
            assert_eq!(
                models,
                vec!["claude-sonnet-4-20250514"],
                "expected rule to match for any complexity"
            );
        }
    }

    #[test]
    fn test_rule_no_match_falls_to_default() {
        let rules = vec![RoutingRule {
            task_type: "code_generation".into(),
            complexity: Some("complex".into()),
            preferred_models: vec!["claude-opus-4-20250514".into()],
            fallback_models: vec![],
        }];
        let config = make_config(rules, vec!["claude-sonnet-4-20250514".into()]);
        // Natural language + simple — no rule matches.
        let result = make_result(TaskType::NaturalLanguage, Complexity::Simple);
        let models = match_routing_rules(&result, &config);
        assert_eq!(models, vec!["claude-sonnet-4-20250514"]);
    }

    #[test]
    fn test_rule_first_match_wins() {
        let rules = vec![
            RoutingRule {
                task_type: "code_generation".into(),
                complexity: None, // matches any complexity
                preferred_models: vec!["model-first".into()],
                fallback_models: vec![],
            },
            RoutingRule {
                task_type: "code_generation".into(),
                complexity: Some("complex".into()),
                preferred_models: vec!["model-second".into()],
                fallback_models: vec![],
            },
        ];
        let config = make_config(rules, vec!["default-model".into()]);
        let result = make_result(
            TaskType::CodeGeneration { language: None },
            Complexity::Complex,
        );
        // The first rule (no complexity filter) should win.
        let models = match_routing_rules(&result, &config);
        assert_eq!(models, vec!["model-first"]);
    }
}
