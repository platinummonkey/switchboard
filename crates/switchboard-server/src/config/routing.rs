//! Semantic routing configuration.

use serde::{Deserialize, Serialize};

fn default_classifier() -> String {
    "heuristic".into()
}

/// A single semantic routing rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Task type to match: "code_generation" | "code_review" | "natural_language" |
    /// "reasoning" | "creative_writing" | "data_analysis" | "tool_use".
    pub task_type: String,

    /// Complexity to match: "simple" | "medium" | "complex". Absent means match any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complexity: Option<String>,

    /// Ordered list of preferred model names for this rule.
    #[serde(default)]
    pub preferred_models: Vec<String>,

    /// Fallback models if preferred are unavailable.
    #[serde(default)]
    pub fallback_models: Vec<String>,
}

/// Default routing when no rule matches.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DefaultRouting {
    #[serde(default)]
    pub preferred_models: Vec<String>,
}

/// Semantic routing configuration (under `[routing.semantic]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRoutingConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Classifier implementation: "heuristic" | "embedding".
    #[serde(default = "default_classifier")]
    pub classifier: String,

    #[serde(default)]
    pub rules: Vec<RoutingRule>,

    #[serde(default)]
    pub default: DefaultRouting,
}

impl Default for SemanticRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            classifier: default_classifier(),
            rules: Vec::new(),
            default: DefaultRouting::default(),
        }
    }
}

/// Top-level routing config (wraps semantic for future extensibility).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RoutingConfig {
    #[serde(default)]
    pub semantic: SemanticRoutingConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_semantic_routing() {
        let toml = r#"
[semantic]
enabled = true
classifier = "heuristic"

[[semantic.rules]]
task_type = "code_generation"
complexity = "complex"
preferred_models = ["claude-opus-4-20250514", "o3"]
fallback_models = ["claude-sonnet-4-20250514"]

[[semantic.rules]]
task_type = "tool_use"
preferred_models = ["claude-sonnet-4-20250514"]

[semantic.default]
preferred_models = ["claude-sonnet-4-20250514"]
"#;
        let c: RoutingConfig = toml::from_str(toml).unwrap();
        assert!(c.semantic.enabled);
        assert_eq!(c.semantic.rules.len(), 2);
        assert_eq!(c.semantic.rules[0].task_type, "code_generation");
        assert_eq!(c.semantic.rules[0].complexity.as_deref(), Some("complex"));
        assert_eq!(
            c.semantic.rules[0].preferred_models[0],
            "claude-opus-4-20250514"
        );
        assert_eq!(c.semantic.rules[1].complexity, None);
        assert_eq!(
            c.semantic.default.preferred_models[0],
            "claude-sonnet-4-20250514"
        );
    }

    #[test]
    fn test_disabled_by_default() {
        let c = SemanticRoutingConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.classifier, "heuristic");
    }

    #[test]
    fn test_parse_disabled() {
        let toml = r#"
[semantic]
enabled = false
classifier = "heuristic"
"#;
        let c: RoutingConfig = toml::from_str(toml).unwrap();
        assert!(!c.semantic.enabled);
        assert!(c.semantic.rules.is_empty());
    }
}
