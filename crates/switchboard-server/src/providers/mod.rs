//! Upstream provider registry and implementations.
//!
//! # Built-in providers
//!
//! - [`AnthropicProvider`] — native Anthropic Messages API
//! - [`OpenAiProvider`] — OpenAI and OpenAI-compatible APIs
//! - [`BedrockProvider`] — Amazon Bedrock Converse API (SigV4 signed)
//! - [`VertexProvider`] — Google Vertex AI (Gemini)
//! - [`OllamaProvider`] — Ollama (OpenAI-compatible local LLM server)
//!
//! # Registry
//!
//! [`ProviderRegistry`] maps provider names to [`UpstreamProvider`] trait objects.
//! Call [`ProviderRegistry::resolve_provider`] to find the right provider for a
//! given model name, consulting the [`ProvidersConfig`] for the `models` list.

pub mod anthropic;
pub mod bedrock;
pub mod ollama;
pub mod openai;
pub mod vertex;

pub use anthropic::AnthropicProvider;
pub use bedrock::BedrockProvider;
pub use ollama::OllamaProvider;
pub use openai::OpenAiProvider;
pub use vertex::VertexProvider;

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::provider::ProvidersConfig;
use crate::routing::{ProviderHealthChecker, UpstreamProvider};

// ── ProviderRegistry ──────────────────────────────────────────────────────────

/// Registry mapping provider names to boxed [`UpstreamProvider`] instances.
///
/// Provider names are short identifiers like `"anthropic"` or `"openai"` that
/// match the keys in the [`ProvidersConfig`] map.
///
/// An optional [`ProviderHealthChecker`] can be attached via
/// [`ProviderRegistry::set_health_checker`].  When present, unhealthy providers
/// are skipped in [`ProviderRegistry::resolve_provider`].  Providers with no
/// recorded health data are treated as healthy (fail-open semantics on startup).
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn UpstreamProvider>>,
    /// Optional health checker; populated after construction via
    /// [`ProviderRegistry::set_health_checker`] once the background task is
    /// running.
    health_checker: Option<Arc<ProviderHealthChecker>>,
}

impl std::fmt::Debug for ProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.providers.keys().map(|s| s.as_str()).collect();
        f.debug_struct("ProviderRegistry")
            .field("providers", &names)
            .finish()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    /// Create an empty registry with no health checker attached.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            health_checker: None,
        }
    }

    /// Attach a [`ProviderHealthChecker`] so that unhealthy providers are
    /// filtered out during [`ProviderRegistry::resolve_provider`].
    ///
    /// This should be called once, after the background health-check task is
    /// spawned, before the registry is wrapped in [`Arc`].
    pub fn set_health_checker(&mut self, checker: Arc<ProviderHealthChecker>) {
        self.health_checker = Some(checker);
    }

    /// Register a provider under `name`.
    ///
    /// If a provider with the same name already exists it is replaced.
    pub fn register(&mut self, name: &str, provider: Arc<dyn UpstreamProvider>) {
        self.providers.insert(name.to_string(), provider);
    }

    /// Look up a provider by its short name (e.g. `"anthropic"`).
    pub fn get(&self, name: &str) -> Option<Arc<dyn UpstreamProvider>> {
        self.providers.get(name).cloned()
    }

    /// Return an iterator over all registered `(name, provider)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Arc<dyn UpstreamProvider>)> {
        self.providers.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Find the provider that can serve `model`, consulting `providers_config`
    /// for the model-to-provider mapping.
    ///
    /// Resolution order:
    /// 1. Walk the providers config entries in insertion order.
    /// 2. For each entry, check whether `model` appears in the `models` list.
    /// 3. If found, look up the runtime provider by the config key name.
    /// 4. Returns the first match as `(provider, resolved_model)`.
    ///
    /// Unhealthy providers (as tracked by the attached [`ProviderHealthChecker`])
    /// are skipped.  If no health checker is configured, all providers are
    /// considered healthy (fail-open).
    ///
    /// The `resolved_model` is the same as `model` — providers are responsible
    /// for their own model-ID translation if needed.
    pub fn resolve_provider(
        &self,
        model: &str,
        providers_config: &ProvidersConfig,
    ) -> Option<(Arc<dyn UpstreamProvider>, String)> {
        // First: try to find a config entry that lists this model.
        for (name, cfg) in providers_config {
            if cfg.models.iter().any(|m| m == model) {
                if let Some(provider) = self.providers.get(name) {
                    if !self.is_provider_healthy(name) {
                        tracing::debug!(
                            model = model,
                            provider = name,
                            "skipping unhealthy provider (config models list)"
                        );
                        continue;
                    }
                    tracing::debug!(
                        model = model,
                        provider = name,
                        "resolved provider via config models list"
                    );
                    return Some((Arc::clone(provider), model.to_string()));
                }
            }
        }

        // Second: fall back to asking each registered provider directly.
        for (name, provider) in &self.providers {
            if provider.supports_model(model) {
                if !self.is_provider_healthy(name) {
                    tracing::debug!(
                        model = model,
                        provider = name,
                        "skipping unhealthy provider (supports_model)"
                    );
                    continue;
                }
                tracing::debug!(
                    model = model,
                    provider = name,
                    "resolved provider via supports_model"
                );
                return Some((Arc::clone(provider), model.to_string()));
            }
        }

        None
    }

    /// Returns `true` if the provider is healthy according to the attached
    /// health checker.  If no health checker is configured, or if the provider
    /// has no recorded health data yet, the result is `true` (fail-open).
    fn is_provider_healthy(&self, name: &str) -> bool {
        match &self.health_checker {
            Some(checker) => checker.is_healthy(name),
            None => true,
        }
    }

    /// Number of registered providers.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Returns `true` if no providers are registered.
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::config::provider::{KeyPoolConfig, ProviderConfig};

    fn make_config(models: Vec<String>, api_format: &str) -> ProviderConfig {
        ProviderConfig {
            base_url: Some("https://example.com".into()),
            api_format: api_format.to_string(),
            models,
            region: None,
            cross_region_inference: false,
            project_id: None,
            timeout: "30s".into(),
            health_check_interval: "30s".into(),
            max_concurrent: 10,
            key_pool: KeyPoolConfig::default(),
        }
    }

    fn make_openai_provider(models: Vec<String>) -> Arc<dyn UpstreamProvider> {
        Arc::new(OpenAiProvider::new_named(
            "openai",
            "https://api.openai.com",
            models,
            Duration::from_secs(30),
        ))
    }

    fn make_anthropic_provider(models: Vec<String>) -> Arc<dyn UpstreamProvider> {
        Arc::new(AnthropicProvider::new_with_base_url(
            "https://api.anthropic.com",
            models,
            Duration::from_secs(30),
        ))
    }

    #[test]
    fn test_registry_new_is_empty() {
        let r = ProviderRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut r = ProviderRegistry::new();
        let p = make_openai_provider(vec!["gpt-4o".into()]);
        r.register("openai", p);
        assert!(r.get("openai").is_some());
        assert!(r.get("anthropic").is_none());
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_registry_register_replaces_existing() {
        let mut r = ProviderRegistry::new();
        let p1 = make_openai_provider(vec!["gpt-4o".into()]);
        let p2 = make_openai_provider(vec!["gpt-4o-mini".into()]);
        r.register("openai", p1);
        r.register("openai", p2);
        assert_eq!(r.len(), 1);
        // The replacement supports gpt-4o-mini.
        let got = r.get("openai").unwrap();
        assert!(got.supports_model("gpt-4o-mini"));
    }

    #[test]
    fn test_resolve_provider_via_config() {
        let mut r = ProviderRegistry::new();
        r.register(
            "openai",
            make_openai_provider(vec!["gpt-4o".into(), "gpt-4o-mini".into()]),
        );
        r.register(
            "anthropic",
            make_anthropic_provider(vec!["claude-sonnet-4-20250514".into()]),
        );

        let mut config = ProvidersConfig::new();
        config.insert(
            "openai".into(),
            make_config(vec!["gpt-4o".into(), "gpt-4o-mini".into()], "openai"),
        );
        config.insert(
            "anthropic".into(),
            make_config(vec!["claude-sonnet-4-20250514".into()], "anthropic"),
        );

        let (provider, model) = r.resolve_provider("gpt-4o", &config).unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(model, "gpt-4o");

        let (provider2, model2) = r
            .resolve_provider("claude-sonnet-4-20250514", &config)
            .unwrap();
        assert_eq!(provider2.name(), "anthropic");
        assert_eq!(model2, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_resolve_provider_not_found() {
        let r = ProviderRegistry::new();
        let config = ProvidersConfig::new();
        assert!(r.resolve_provider("unknown-model", &config).is_none());
    }

    #[test]
    fn test_resolve_provider_fallback_to_supports_model() {
        let mut r = ProviderRegistry::new();
        // Register a provider that supports gpt-4o but config lists no models.
        r.register("openai", make_openai_provider(vec!["gpt-4o".into()]));

        // Config has no models listed for openai.
        let mut config = ProvidersConfig::new();
        config.insert("openai".into(), make_config(vec![], "openai"));

        // Should fall back to supports_model().
        let result = r.resolve_provider("gpt-4o", &config);
        assert!(result.is_some());
        let (provider, _) = result.unwrap();
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_registry_iter() {
        let mut r = ProviderRegistry::new();
        r.register("openai", make_openai_provider(vec![]));
        r.register("anthropic", make_anthropic_provider(vec![]));
        let names: Vec<&str> = r.iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"openai"));
        assert!(names.contains(&"anthropic"));
    }

    #[test]
    fn test_resolve_provider_skips_unhealthy_via_config() {
        let mut r = ProviderRegistry::new();
        r.register("openai", make_openai_provider(vec!["gpt-4o".into()]));

        // Mark openai as unhealthy.
        let checker = Arc::new(ProviderHealthChecker::new());
        checker.record("openai", false);
        r.set_health_checker(Arc::clone(&checker));

        let mut config = ProvidersConfig::new();
        config.insert(
            "openai".into(),
            make_config(vec!["gpt-4o".into()], "openai"),
        );

        // Should return None because the only matching provider is unhealthy.
        assert!(
            r.resolve_provider("gpt-4o", &config).is_none(),
            "unhealthy provider must not be returned"
        );
    }

    #[test]
    fn test_resolve_provider_skips_unhealthy_via_supports_model() {
        let mut r = ProviderRegistry::new();
        r.register("openai", make_openai_provider(vec!["gpt-4o".into()]));

        // Mark openai as unhealthy.
        let checker = Arc::new(ProviderHealthChecker::new());
        checker.record("openai", false);
        r.set_health_checker(Arc::clone(&checker));

        // Empty config forces the fallback supports_model() path.
        let config = ProvidersConfig::new();

        assert!(
            r.resolve_provider("gpt-4o", &config).is_none(),
            "unhealthy provider must not be returned via supports_model path"
        );
    }

    #[test]
    fn test_resolve_provider_returns_healthy_provider() {
        let mut r = ProviderRegistry::new();
        r.register("openai", make_openai_provider(vec!["gpt-4o".into()]));

        // Mark openai as healthy explicitly.
        let checker = Arc::new(ProviderHealthChecker::new());
        checker.record("openai", true);
        r.set_health_checker(Arc::clone(&checker));

        let mut config = ProvidersConfig::new();
        config.insert(
            "openai".into(),
            make_config(vec!["gpt-4o".into()], "openai"),
        );

        let result = r.resolve_provider("gpt-4o", &config);
        assert!(result.is_some(), "healthy provider must be returned");
        let (provider, _) = result.unwrap();
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_resolve_provider_no_checker_is_fail_open() {
        // No health checker attached — should behave like old code (always healthy).
        let mut r = ProviderRegistry::new();
        r.register("openai", make_openai_provider(vec!["gpt-4o".into()]));

        let mut config = ProvidersConfig::new();
        config.insert(
            "openai".into(),
            make_config(vec!["gpt-4o".into()], "openai"),
        );

        let result = r.resolve_provider("gpt-4o", &config);
        assert!(
            result.is_some(),
            "without health checker all providers are healthy"
        );
    }
}
