//! Model selection policies.
//!
//! [`ModelSelector`] evaluates the configured [`ModelSelectionConfig`] and
//! a [`RequestContext`] to produce a `(model_name, SelectionReason)` pair.

use switchboard_common::types::RequestContext;

use crate::config::model_selection::ModelSelectionConfig;
use crate::routing::SelectionReason;

// ── Glob matching ─────────────────────────────────────────────────────────────

/// Returns `true` when `value` matches `pattern`.
///
/// The only wildcard supported is `*`, which matches zero or more of any
/// character (similar to shell glob, but without path separator semantics).
///
/// Examples:
/// - `"claude-*"` matches `"claude-sonnet-4-20250514"`
/// - `"gpt-4*"` matches `"gpt-4o"` and `"gpt-4o-mini"`
/// - `"claude-sonnet-4-20250514"` matches only that exact string
pub fn glob_match(pattern: &str, value: &str) -> bool {
    let pattern = pattern.as_bytes();
    let value = value.as_bytes();
    glob_match_bytes(pattern, value)
}

fn glob_match_bytes(pattern: &[u8], value: &[u8]) -> bool {
    match (pattern.first(), value.first()) {
        // Both exhausted: full match.
        (None, None) => true,
        // Pattern exhausted but value still has chars: no match.
        (None, Some(_)) => false,
        // Wildcard: try matching zero characters (advance pattern only) or
        // one character (advance value only, keep wildcard to consume more).
        (Some(&b'*'), _) => {
            // Skip consecutive wildcards.
            glob_match_bytes(&pattern[1..], value)
                || (!value.is_empty() && glob_match_bytes(pattern, &value[1..]))
        }
        // Value exhausted but pattern still has non-wildcard chars: no match.
        (Some(_), None) => false,
        // Literal match: advance both.
        (Some(&p), Some(&v)) if p == v => glob_match_bytes(&pattern[1..], &value[1..]),
        // Character mismatch: no match.
        _ => false,
    }
}

// ── ModelSelector ─────────────────────────────────────────────────────────────

/// Evaluates the configured [`ModelSelectionConfig`] and a [`RequestContext`]
/// to produce a `(model_name, SelectionReason)` pair.
///
/// # Modes
///
/// | Mode       | Behaviour |
/// |------------|-----------|
/// | `static`   | Always returns the configured `model` (or `fallback`). |
/// | `mapping`  | Looks up `ctx.model` in `config.mappings`; falls through to `fallback`. |
/// | `dynamic`  | Honours a client-supplied model header, per-user/team overrides, then `fallback`. |
/// | `semantic` | Defers to [`crate::routing::SemanticClassifier`] (async); this method returns `fallback` synchronously. The caller is responsible for running the async classifier and substituting the result before forwarding the request. |
pub struct ModelSelector {
    config: ModelSelectionConfig,
}

impl ModelSelector {
    /// Create a new selector from the given configuration.
    pub fn new(config: ModelSelectionConfig) -> Self {
        Self { config }
    }

    /// Select a model for the given request context.
    ///
    /// Returns `(model_name, reason)`.
    pub fn select(&self, ctx: &RequestContext) -> (String, SelectionReason) {
        match self.config.mode.as_str() {
            "static" => self.select_static(),
            "mapping" => self.select_mapping(ctx),
            "dynamic" => self.select_dynamic(ctx),
            "semantic" => {
                // Semantic mode delegates to an async SemanticClassifier.
                // This synchronous method returns the configured fallback so
                // the proxy layer can use it before the async path completes.
                // The caller MUST run the classifier and replace this result.
                tracing::debug!(
                    "model_selection mode=semantic: returning fallback synchronously; \
                     caller must run async classifier"
                );
                (self.fallback(), SelectionReason::Fallback)
            }
            unknown => {
                tracing::warn!(
                    mode = unknown,
                    "unknown model_selection mode, falling back to fallback model"
                );
                (self.fallback(), SelectionReason::Fallback)
            }
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Return the fallback model name, using a hardcoded default when none is
    /// configured.
    fn fallback(&self) -> String {
        self.config
            .fallback
            .clone()
            .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string())
    }

    fn select_static(&self) -> (String, SelectionReason) {
        let model = self.config.model.clone().unwrap_or_else(|| self.fallback());
        (model, SelectionReason::Static)
    }

    fn select_mapping(&self, ctx: &RequestContext) -> (String, SelectionReason) {
        if let Some(requested) = &ctx.model {
            if let Some(mapped) = self.config.mappings.get(requested.as_str()) {
                tracing::debug!(
                    requested = requested.as_str(),
                    mapped = mapped.as_str(),
                    "model_selection: mapping hit"
                );
                return (mapped.clone(), SelectionReason::Mapping);
            }
        }
        tracing::debug!("model_selection: mapping miss, using fallback");
        (self.fallback(), SelectionReason::Fallback)
    }

    fn select_dynamic(&self, ctx: &RequestContext) -> (String, SelectionReason) {
        // 1. Check for a client-supplied model header.
        if let Some(header_model) = ctx.switchboard_headers.get(&self.config.header) {
            if self.is_allowed(header_model) {
                tracing::debug!(
                    model = header_model.as_str(),
                    "model_selection: header override accepted"
                );
                return (header_model.clone(), SelectionReason::HeaderOverride);
            }
            tracing::debug!(
                model = header_model.as_str(),
                "model_selection: header override rejected (not in allowed_models)"
            );
        }

        // 2. Per-user override.
        if let Some(user_id) = &ctx.user_id {
            if let Some(override_model) = self.config.overrides.get(user_id.as_str()) {
                tracing::debug!(
                    user_id = user_id.as_str(),
                    model = override_model.as_str(),
                    "model_selection: user override applied"
                );
                return (override_model.clone(), SelectionReason::UserOverride);
            }
        }

        // 3. Per-team override.
        if let Some(team) = &ctx.team {
            if let Some(override_model) = self.config.overrides.get(team.as_str()) {
                tracing::debug!(
                    team = team.as_str(),
                    model = override_model.as_str(),
                    "model_selection: team override applied"
                );
                return (override_model.clone(), SelectionReason::UserOverride);
            }
        }

        // 4. Fallback.
        tracing::debug!("model_selection: no dynamic rule matched, using fallback");
        (self.fallback(), SelectionReason::Fallback)
    }

    /// Returns `true` when `model` is permitted by the `allowed_models` list.
    ///
    /// An empty list means *all* models are allowed (no restriction).
    fn is_allowed(&self, model: &str) -> bool {
        if self.config.allowed_models.is_empty() {
            return true;
        }
        self.config
            .allowed_models
            .iter()
            .any(|pattern| glob_match(pattern, model))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::config::model_selection::ModelSelectionConfig;
    use switchboard_common::types::RequestContext;

    fn ctx_with_model(model: &str) -> RequestContext {
        RequestContext {
            model: Some(model.into()),
            ..RequestContext::new()
        }
    }

    fn ctx_with_header(header_key: &str, header_val: &str) -> RequestContext {
        let mut headers = HashMap::new();
        headers.insert(header_key.to_string(), header_val.to_string());
        RequestContext {
            switchboard_headers: headers,
            ..RequestContext::new()
        }
    }

    fn ctx_with_user(user_id: &str) -> RequestContext {
        RequestContext {
            user_id: Some(user_id.into()),
            ..RequestContext::new()
        }
    }

    // ── glob_match ────────────────────────────────────────────────────────────

    #[test]
    fn test_glob_match_star_wildcard() {
        assert!(glob_match("claude-*", "claude-sonnet-4-20250514"));
        assert!(glob_match("claude-*", "claude-opus-4-20250514"));
        assert!(glob_match("claude-*", "claude-haiku-4-5-20251001"));
        assert!(glob_match("gpt-*", "gpt-4o"));
        assert!(glob_match("gpt-*", "gpt-4o-mini"));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn test_glob_match_exact() {
        assert!(glob_match(
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-20250514"
        ));
        assert!(glob_match("gpt-4o", "gpt-4o"));
    }

    #[test]
    fn test_glob_match_no_match() {
        assert!(!glob_match("claude-*", "gpt-4o"));
        assert!(!glob_match(
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514"
        ));
        assert!(!glob_match("gpt-4o", "gpt-4o-mini"));
    }

    // ── static mode ───────────────────────────────────────────────────────────

    #[test]
    fn test_static_mode_always_returns_configured_model() {
        let config = ModelSelectionConfig {
            mode: "static".into(),
            model: Some("claude-opus-4-20250514".into()),
            fallback: Some("claude-sonnet-4-20250514".into()),
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = RequestContext::new();
        let (model, reason) = selector.select(&ctx);
        assert_eq!(model, "claude-opus-4-20250514");
        assert!(matches!(reason, SelectionReason::Static));
    }

    #[test]
    fn test_static_mode_uses_fallback_when_no_model_set() {
        let config = ModelSelectionConfig {
            mode: "static".into(),
            model: None,
            fallback: Some("claude-haiku-4-5-20251001".into()),
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = RequestContext::new();
        let (model, reason) = selector.select(&ctx);
        assert_eq!(model, "claude-haiku-4-5-20251001");
        assert!(matches!(reason, SelectionReason::Static));
    }

    // ── mapping mode ──────────────────────────────────────────────────────────

    #[test]
    fn test_mapping_mode_maps_known_model() {
        let mut mappings = HashMap::new();
        mappings.insert("gpt-4".to_string(), "claude-sonnet-4-20250514".to_string());
        let config = ModelSelectionConfig {
            mode: "mapping".into(),
            fallback: Some("claude-sonnet-4-20250514".into()),
            mappings,
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = ctx_with_model("gpt-4");
        let (model, reason) = selector.select(&ctx);
        assert_eq!(model, "claude-sonnet-4-20250514");
        assert!(matches!(reason, SelectionReason::Mapping));
    }

    #[test]
    fn test_mapping_mode_falls_back_for_unknown() {
        let config = ModelSelectionConfig {
            mode: "mapping".into(),
            fallback: Some("claude-sonnet-4-20250514".into()),
            mappings: HashMap::new(),
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = ctx_with_model("unknown-model");
        let (model, reason) = selector.select(&ctx);
        assert_eq!(model, "claude-sonnet-4-20250514");
        assert!(matches!(reason, SelectionReason::Fallback));
    }

    // ── dynamic mode ──────────────────────────────────────────────────────────

    #[test]
    fn test_dynamic_mode_uses_header() {
        let config = ModelSelectionConfig {
            mode: "dynamic".into(),
            header: "x-switchboard-model".into(),
            fallback: Some("claude-sonnet-4-20250514".into()),
            allowed_models: vec!["claude-*".into()],
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = ctx_with_header("x-switchboard-model", "claude-opus-4-20250514");
        let (model, reason) = selector.select(&ctx);
        assert_eq!(model, "claude-opus-4-20250514");
        assert!(matches!(reason, SelectionReason::HeaderOverride));
    }

    #[test]
    fn test_dynamic_mode_glob_allows_claude_star() {
        let config = ModelSelectionConfig {
            mode: "dynamic".into(),
            header: "x-switchboard-model".into(),
            fallback: Some("claude-sonnet-4-20250514".into()),
            allowed_models: vec!["claude-*".into()],
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = ctx_with_header("x-switchboard-model", "claude-haiku-4-5-20251001");
        let (model, reason) = selector.select(&ctx);
        assert_eq!(model, "claude-haiku-4-5-20251001");
        assert!(matches!(reason, SelectionReason::HeaderOverride));
    }

    #[test]
    fn test_dynamic_mode_glob_rejects_gpt_when_only_claude_allowed() {
        let config = ModelSelectionConfig {
            mode: "dynamic".into(),
            header: "x-switchboard-model".into(),
            fallback: Some("claude-sonnet-4-20250514".into()),
            allowed_models: vec!["claude-*".into()],
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = ctx_with_header("x-switchboard-model", "gpt-4o");
        let (model, reason) = selector.select(&ctx);
        // gpt-4o rejected, falls back.
        assert_eq!(model, "claude-sonnet-4-20250514");
        assert!(matches!(reason, SelectionReason::Fallback));
    }

    #[test]
    fn test_dynamic_mode_user_override() {
        let mut overrides = HashMap::new();
        overrides.insert("alice".to_string(), "claude-opus-4-20250514".to_string());
        let config = ModelSelectionConfig {
            mode: "dynamic".into(),
            header: "x-switchboard-model".into(),
            fallback: Some("claude-sonnet-4-20250514".into()),
            overrides,
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = ctx_with_user("alice");
        let (model, reason) = selector.select(&ctx);
        assert_eq!(model, "claude-opus-4-20250514");
        assert!(matches!(reason, SelectionReason::UserOverride));
    }

    #[test]
    fn test_dynamic_mode_fallback_when_no_header() {
        let config = ModelSelectionConfig {
            mode: "dynamic".into(),
            header: "x-switchboard-model".into(),
            fallback: Some("claude-sonnet-4-20250514".into()),
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = RequestContext::new();
        let (model, reason) = selector.select(&ctx);
        assert_eq!(model, "claude-sonnet-4-20250514");
        assert!(matches!(reason, SelectionReason::Fallback));
    }

    // ── semantic mode ─────────────────────────────────────────────────────────

    #[test]
    fn test_semantic_mode_returns_fallback_synchronously() {
        let config = ModelSelectionConfig {
            mode: "semantic".into(),
            fallback: Some("claude-sonnet-4-20250514".into()),
            ..ModelSelectionConfig::default()
        };
        let selector = ModelSelector::new(config);
        let ctx = RequestContext::new();
        let (model, reason) = selector.select(&ctx);
        assert_eq!(model, "claude-sonnet-4-20250514");
        assert!(matches!(reason, SelectionReason::Fallback));
    }
}
