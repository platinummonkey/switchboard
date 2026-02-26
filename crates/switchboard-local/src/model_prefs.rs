//! Model preference injection for `switchboard-local`.
//!
//! [`ModelPrefs`] reads the `[model]` section of the local config and provides:
//! - A default model to inject when the client request does not specify one.
//! - An override table that rewrites specific model names (e.g. redirect
//!   `"gpt-4"` → `"claude-sonnet-4-20250514"` transparently).

use std::collections::HashMap;

use crate::config::ModelConfig;

/// Holds the default model and override mappings loaded from [`ModelConfig`].
#[derive(Debug, Clone)]
pub struct ModelPrefs {
    default: String,
    overrides: HashMap<String, String>,
}

impl ModelPrefs {
    /// Build a [`ModelPrefs`] from the `[model]` config section.
    pub fn from_config(config: &ModelConfig) -> Self {
        Self {
            default: config.default.clone(),
            overrides: config.overrides.clone(),
        }
    }

    /// Return the default model name to inject when the client does not supply
    /// a `model` field.
    pub fn default_model(&self) -> &str {
        &self.default
    }

    /// If `requested` has a configured override, return the override model name.
    /// Otherwise return `None` (meaning the requested model passes through
    /// unchanged).
    pub fn override_model<'a>(&'a self, requested: &'a str) -> Option<&'a str> {
        self.overrides.get(requested).map(String::as_str)
    }

    /// Rewrite the `"model"` field of `body` in place if an override mapping
    /// exists for the current value.
    ///
    /// If the body has no `"model"` field, or there is no override for the
    /// current value, the body is left unchanged.
    pub fn apply_to_body(&self, body: &mut serde_json::Value) {
        if let Some(model_val) = body.get("model") {
            if let Some(current) = model_val.as_str() {
                if let Some(replacement) = self.override_model(current) {
                    body["model"] = serde_json::Value::String(replacement.to_owned());
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn make_prefs() -> ModelPrefs {
        let mut overrides = HashMap::new();
        overrides.insert("gpt-4".into(), "claude-sonnet-4-20250514".into());
        overrides.insert("gpt-4o".into(), "claude-opus-4-20250514".into());
        ModelPrefs {
            default: "claude-sonnet-4-20250514".into(),
            overrides,
        }
    }

    #[test]
    fn test_model_prefs_default() {
        let prefs = make_prefs();
        assert_eq!(prefs.default_model(), "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_model_prefs_override_known_model() {
        let prefs = make_prefs();
        assert_eq!(
            prefs.override_model("gpt-4"),
            Some("claude-sonnet-4-20250514")
        );
    }

    #[test]
    fn test_model_prefs_override_unknown_model_returns_none() {
        let prefs = make_prefs();
        assert_eq!(prefs.override_model("some-unknown-model"), None);
    }

    #[test]
    fn test_model_prefs_apply_to_body_rewrites_model() {
        let prefs = make_prefs();
        let mut body = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}]
        });
        prefs.apply_to_body(&mut body);
        assert_eq!(body["model"].as_str(), Some("claude-sonnet-4-20250514"));
    }

    #[test]
    fn test_model_prefs_apply_to_body_no_override_leaves_unchanged() {
        let prefs = make_prefs();
        let mut body = serde_json::json!({
            "model": "custom-model",
            "messages": []
        });
        prefs.apply_to_body(&mut body);
        assert_eq!(body["model"].as_str(), Some("custom-model"));
    }

    #[test]
    fn test_model_prefs_apply_to_body_no_model_field_leaves_unchanged() {
        let prefs = make_prefs();
        let mut body = serde_json::json!({ "messages": [] });
        let original = body.clone();
        prefs.apply_to_body(&mut body);
        assert_eq!(body, original);
    }

    #[test]
    fn test_model_prefs_from_config() {
        let mut overrides = HashMap::new();
        overrides.insert("gpt-4".into(), "claude-sonnet-4-20250514".into());
        let config = ModelConfig {
            default: "claude-sonnet-4-20250514".into(),
            overrides,
        };
        let prefs = ModelPrefs::from_config(&config);
        assert_eq!(prefs.default_model(), "claude-sonnet-4-20250514");
        assert_eq!(
            prefs.override_model("gpt-4"),
            Some("claude-sonnet-4-20250514")
        );
    }
}
