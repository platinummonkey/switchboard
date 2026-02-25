//! Model selection policy configuration.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

fn default_mode() -> String {
    "dynamic".into()
}

fn default_header() -> String {
    switchboard_common::protocol::HEADER_MODEL.into()
}

/// How the server selects which model to use for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionConfig {
    /// Selection mode: "static" | "mapping" | "dynamic" | "semantic".
    #[serde(default = "default_mode")]
    pub mode: String,

    /// Fixed model to always use (mode = "static").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Header clients use to request a model (mode = "dynamic").
    #[serde(default = "default_header")]
    pub header: String,

    /// Model to use when no preference is indicated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,

    /// Model name mappings (mode = "mapping"): requested → actual.
    #[serde(default)]
    pub mappings: HashMap<String, String>,

    /// Glob patterns of models clients are allowed to request.
    /// Empty list means all models are allowed.
    #[serde(default)]
    pub allowed_models: Vec<String>,

    /// Per-user or per-team model overrides, keyed by user/team ID.
    #[serde(default)]
    pub overrides: HashMap<String, String>,
}

impl Default for ModelSelectionConfig {
    fn default() -> Self {
        Self {
            mode: default_mode(),
            model: None,
            header: default_header(),
            fallback: Some("claude-sonnet-4-20250514".into()),
            mappings: HashMap::new(),
            allowed_models: Vec::new(),
            overrides: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dynamic_mode() {
        let toml = r#"
mode = "dynamic"
header = "x-switchboard-model"
fallback = "claude-sonnet-4-20250514"
"#;
        let c: ModelSelectionConfig = toml::from_str(toml).unwrap();
        assert_eq!(c.mode, "dynamic");
        assert_eq!(c.fallback.as_deref(), Some("claude-sonnet-4-20250514"));
    }

    #[test]
    fn test_parse_static_mode() {
        let toml = r#"mode = "static"
model = "claude-opus-4-20250514"
"#;
        let c: ModelSelectionConfig = toml::from_str(toml).unwrap();
        assert_eq!(c.mode, "static");
        assert_eq!(c.model.as_deref(), Some("claude-opus-4-20250514"));
    }

    #[test]
    fn test_parse_mapping_mode() {
        let toml = r#"
mode = "mapping"
[mappings]
"gpt-4" = "claude-sonnet-4-20250514"
"gpt-3.5-turbo" = "claude-haiku-4-5-20251001"
"#;
        let c: ModelSelectionConfig = toml::from_str(toml).unwrap();
        assert_eq!(c.mode, "mapping");
        assert_eq!(c.mappings["gpt-4"], "claude-sonnet-4-20250514");
        assert_eq!(c.mappings.len(), 2);
    }

    #[test]
    fn test_defaults() {
        let c = ModelSelectionConfig::default();
        assert_eq!(c.mode, "dynamic");
        assert!(c.mappings.is_empty());
    }
}
