//! Model-selection override Tower layer.
//!
//! This layer runs **before** the proxy handler.  It examines each incoming
//! request, applies the configured [`ModelSelector`] policy to determine which
//! model (and therefore which upstream provider) should handle it, and stores
//! the results in request extensions so the handler and the
//! [`super::auth_inject::AuthInjectLayer`] can use them without repeating the
//! resolution logic.
//!
//! # Body-peeking strategy
//!
//! Reading the request body in Tower middleware would consume the `Body` stream
//! before the axum handler can read it.  To avoid that problem, the layer reads
//! the model name from the **`x-switchboard-model` header** (injected by
//! `switchboard-local` or sent directly by callers).  If no explicit override
//! header is present, it falls back to calling `selector.select(&ctx)` with a
//! minimal [`RequestContext`] so the configured policy (static, mapping,
//! dynamic, fallback) applies.
//!
//! # Extensions set
//!
//! | Type | Description |
//! |------|-------------|
//! | [`ResolvedModel`] | Final model name after policy evaluation |
//! | [`ProviderName`] | Short provider name (e.g. `"anthropic"`) |
//! | [`SelectionReasonExt`] | Why this model was chosen |
//!
//! If the registry cannot find a provider for the resolved model, **no
//! extensions are set** and the handler is responsible for returning 400/404.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::Body;
use http::{Request, Response};
use tower::{Layer, Service};

use switchboard_common::protocol::HEADER_MODEL;
use switchboard_common::types::RequestContext;

use crate::config::provider::ProvidersConfig;
use crate::providers::ProviderRegistry;
use crate::routing::{ModelSelector, SelectionReason};

// ── Extension types ───────────────────────────────────────────────────────────

/// The resolved model name after applying the selection policy.
#[derive(Clone, Debug)]
pub struct ResolvedModel(pub String);

/// The provider name that should handle this request (e.g. `"anthropic"`).
#[derive(Clone, Debug)]
pub struct ProviderName(pub String);

/// The reason this model was selected, for observability.
#[derive(Clone, Debug)]
pub struct SelectionReasonExt(pub SelectionReason);

// ── Layer ─────────────────────────────────────────────────────────────────────

/// Tower [`Layer`] that resolves the model and provider for each request and
/// stores the results in extensions.
#[derive(Clone)]
pub struct ModelOverrideLayer {
    selector: Arc<ModelSelector>,
    registry: Arc<ProviderRegistry>,
    providers_config: Arc<ProvidersConfig>,
}

impl std::fmt::Debug for ModelOverrideLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelOverrideLayer")
            .field("registry", &self.registry)
            .finish()
    }
}

impl ModelOverrideLayer {
    /// Create a new layer.
    ///
    /// - `selector` — evaluates the model selection policy
    /// - `registry` — maps model names to upstream providers
    /// - `providers_config` — provider configuration used by `resolve_provider`
    pub fn new(
        selector: Arc<ModelSelector>,
        registry: Arc<ProviderRegistry>,
        providers_config: Arc<ProvidersConfig>,
    ) -> Self {
        Self {
            selector,
            registry,
            providers_config,
        }
    }
}

impl<S> Layer<S> for ModelOverrideLayer {
    type Service = ModelOverrideService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ModelOverrideService {
            inner,
            selector: Arc::clone(&self.selector),
            registry: Arc::clone(&self.registry),
            providers_config: Arc::clone(&self.providers_config),
        }
    }
}

// ── Service ───────────────────────────────────────────────────────────────────

/// Tower [`Service`] produced by [`ModelOverrideLayer`].
#[derive(Clone)]
pub struct ModelOverrideService<S> {
    inner: S,
    selector: Arc<ModelSelector>,
    registry: Arc<ProviderRegistry>,
    providers_config: Arc<ProvidersConfig>,
}

impl<S> std::fmt::Debug for ModelOverrideService<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelOverrideService")
            .finish_non_exhaustive()
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

impl<S> Service<Request<Body>> for ModelOverrideService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let selector = Arc::clone(&self.selector);
        let registry = Arc::clone(&self.registry);
        let providers_config = Arc::clone(&self.providers_config);
        let mut inner = self.inner.clone();

        Box::pin(async move {
            // Step 1: determine the model name.
            //
            // Priority order:
            //   (a) `x-switchboard-model` header — explicit client override
            //   (b) `selector.select()` — applies configured policy (static /
            //       mapping / dynamic / fallback)
            let (resolved_model, reason) = if let Some(header_model) = req
                .headers()
                .get(HEADER_MODEL)
                .and_then(|v| v.to_str().ok())
            {
                tracing::debug!(
                    model = header_model,
                    "model_override: model from x-switchboard-model header"
                );
                (header_model.to_owned(), SelectionReason::HeaderOverride)
            } else {
                // Build a minimal RequestContext; full user context is not
                // yet available at this layer (AuthLayer runs after us in
                // the stack ordering described in mod.rs, but in practice
                // ModelOverrideLayer is outermost after RequestIdLayer so
                // ValidatedClient may not be set yet).
                let ctx = RequestContext::default();
                let (model, reason) = selector.select(&ctx);
                tracing::debug!(
                    model = %model,
                    reason = %reason,
                    "model_override: model from selector policy"
                );
                (model, reason)
            };

            // Step 2: resolve the provider for the model.
            //
            // If the registry has no provider for this model we skip setting
            // extensions; the downstream handler will detect the missing
            // ProviderName and return an appropriate error.
            match registry.resolve_provider(&resolved_model, &providers_config) {
                Some((provider, model)) => {
                    let provider_name = provider.name().to_string();
                    tracing::debug!(
                        model = %model,
                        provider = %provider_name,
                        reason = %reason,
                        "model_override: resolved provider"
                    );

                    let mut req = req;
                    req.extensions_mut().insert(ResolvedModel(model));
                    req.extensions_mut().insert(ProviderName(provider_name));
                    req.extensions_mut().insert(SelectionReasonExt(reason));
                    inner.call(req).await
                }
                None => {
                    tracing::debug!(
                        model = %resolved_model,
                        "model_override: no provider found, passing through without extensions"
                    );
                    // Pass through — handler is responsible for the error.
                    inner.call(req).await
                }
            }
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::task::Poll;
    use std::time::Duration;

    use axum::body::Body;
    use http::{Request, Response, StatusCode};
    use tower::{Layer, Service, ServiceExt};

    use super::*;
    use crate::config::model_selection::ModelSelectionConfig;
    use crate::config::provider::{KeyPoolConfig, ProviderConfig, ProvidersConfig};
    use crate::providers::{AnthropicProvider, OpenAiProvider, ProviderRegistry};
    use crate::routing::ModelSelector;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn static_selector(model: &str) -> Arc<ModelSelector> {
        Arc::new(ModelSelector::new(ModelSelectionConfig {
            mode: "static".into(),
            model: Some(model.to_string()),
            fallback: Some(model.to_string()),
            ..ModelSelectionConfig::default()
        }))
    }

    fn registry_with_anthropic(models: Vec<String>) -> Arc<ProviderRegistry> {
        let mut r = ProviderRegistry::new();
        r.register(
            "anthropic",
            Arc::new(AnthropicProvider::new_with_base_url(
                "https://api.anthropic.com",
                models,
                Duration::from_secs(30),
            )),
        );
        Arc::new(r)
    }

    fn registry_with_openai(models: Vec<String>) -> Arc<ProviderRegistry> {
        let mut r = ProviderRegistry::new();
        r.register(
            "openai",
            Arc::new(OpenAiProvider::new_named(
                "openai",
                "https://api.openai.com",
                models,
                Duration::from_secs(30),
            )),
        );
        Arc::new(r)
    }

    fn providers_config_with(name: &str, models: Vec<String>) -> Arc<ProvidersConfig> {
        let mut map: ProvidersConfig = HashMap::new();
        map.insert(
            name.to_string(),
            ProviderConfig {
                base_url: Some("https://example.com".into()),
                api_format: name.to_string(),
                models,
                region: None,
                cross_region_inference: false,
                project_id: None,
                timeout: "30s".into(),
                health_check_interval: "30s".into(),
                max_concurrent: 10,
                key_pool: KeyPoolConfig::default(),
            },
        );
        Arc::new(map)
    }

    fn empty_registry() -> Arc<ProviderRegistry> {
        Arc::new(ProviderRegistry::new())
    }

    fn empty_providers_config() -> Arc<ProvidersConfig> {
        Arc::new(HashMap::new())
    }

    // A capture service that records the extensions it received.
    #[derive(Clone)]
    struct CaptureSvc {
        resolved_model: Arc<std::sync::Mutex<Option<Option<ResolvedModel>>>>,
        provider_name: Arc<std::sync::Mutex<Option<Option<ProviderName>>>>,
    }

    impl CaptureSvc {
        #[allow(clippy::type_complexity)]
        fn new() -> (
            Self,
            Arc<std::sync::Mutex<Option<Option<ResolvedModel>>>>,
            Arc<std::sync::Mutex<Option<Option<ProviderName>>>>,
        ) {
            let rm = Arc::new(std::sync::Mutex::new(None));
            let pn = Arc::new(std::sync::Mutex::new(None));
            (
                Self {
                    resolved_model: Arc::clone(&rm),
                    provider_name: Arc::clone(&pn),
                },
                rm,
                pn,
            )
        }
    }

    impl Service<Request<Body>> for CaptureSvc {
        type Response = Response<Body>;
        type Error = Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(
            &mut self,
            _cx: &mut std::task::Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, req: Request<Body>) -> Self::Future {
            *self.resolved_model.lock().unwrap() =
                Some(req.extensions().get::<ResolvedModel>().cloned());
            *self.provider_name.lock().unwrap() =
                Some(req.extensions().get::<ProviderName>().cloned());
            std::future::ready(Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::empty())
                .unwrap()))
        }
    }

    // ── Tests: extension type derive impls ────────────────────────────────────

    #[test]
    fn test_resolved_model_clone_and_debug() {
        let m = ResolvedModel("claude-opus".to_string());
        let cloned = m.clone();
        assert_eq!(cloned.0, "claude-opus");
        assert!(format!("{m:?}").contains("claude-opus"));
    }

    #[test]
    fn test_provider_name_clone_and_debug() {
        let p = ProviderName("anthropic".to_string());
        let cloned = p.clone();
        assert_eq!(cloned.0, "anthropic");
        assert!(format!("{p:?}").contains("anthropic"));
    }

    #[test]
    fn test_selection_reason_ext_clone_and_debug() {
        let r = SelectionReasonExt(SelectionReason::Static);
        let cloned = r.clone();
        assert!(matches!(cloned.0, SelectionReason::Static));
        assert!(format!("{r:?}").contains("Static"));
    }

    // ── Tests: model resolution from header ───────────────────────────────────

    #[tokio::test]
    async fn test_resolved_model_from_header() {
        let selector = static_selector("claude-sonnet-4-20250514");
        let registry = registry_with_anthropic(vec!["claude-opus".to_string()]);
        let config = providers_config_with("anthropic", vec!["claude-opus".to_string()]);

        let (svc, rm, _pn) = CaptureSvc::new();
        let layer = ModelOverrideLayer::new(selector, registry, config);
        let mut wrapped = layer.layer(svc);

        let mut req = Request::new(Body::empty());
        req.headers_mut()
            .insert(HEADER_MODEL, "claude-opus".parse().unwrap());

        wrapped.ready().await.unwrap();
        wrapped.call(req).await.unwrap();

        let guard = rm.lock().unwrap();
        let resolved = guard.as_ref().unwrap().as_ref();
        assert!(resolved.is_some(), "ResolvedModel should be set");
        assert_eq!(resolved.unwrap().0, "claude-opus");
    }

    #[tokio::test]
    async fn test_resolved_model_falls_back_to_selector() {
        // No header; selector returns "claude-sonnet-4-20250514"
        let selector = static_selector("claude-sonnet-4-20250514");
        let registry = registry_with_anthropic(vec!["claude-sonnet-4-20250514".to_string()]);
        let config =
            providers_config_with("anthropic", vec!["claude-sonnet-4-20250514".to_string()]);

        let (svc, rm, _pn) = CaptureSvc::new();
        let layer = ModelOverrideLayer::new(selector, registry, config);
        let mut wrapped = layer.layer(svc);

        wrapped.ready().await.unwrap();
        wrapped.call(Request::new(Body::empty())).await.unwrap();

        let guard = rm.lock().unwrap();
        let resolved = guard.as_ref().unwrap().as_ref();
        assert!(resolved.is_some(), "ResolvedModel should be set");
        assert_eq!(resolved.unwrap().0, "claude-sonnet-4-20250514");
    }

    // ── Tests: provider name resolution ───────────────────────────────────────

    #[tokio::test]
    async fn test_provider_name_set_when_registry_matches() {
        let selector = static_selector("gpt-4o");
        let registry = registry_with_openai(vec!["gpt-4o".to_string()]);
        let config = providers_config_with("openai", vec!["gpt-4o".to_string()]);

        let (svc, _rm, pn) = CaptureSvc::new();
        let layer = ModelOverrideLayer::new(selector, registry, config);
        let mut wrapped = layer.layer(svc);

        wrapped.ready().await.unwrap();
        wrapped.call(Request::new(Body::empty())).await.unwrap();

        let guard = pn.lock().unwrap();
        let provider = guard.as_ref().unwrap().as_ref();
        assert!(provider.is_some(), "ProviderName should be set");
        assert_eq!(provider.unwrap().0, "openai");
    }

    #[tokio::test]
    async fn test_provider_name_not_set_when_no_match() {
        // Registry is empty — no provider for any model.
        let selector = static_selector("unknown-model");
        let registry = empty_registry();
        let config = empty_providers_config();

        let (svc, _rm, pn) = CaptureSvc::new();
        let layer = ModelOverrideLayer::new(selector, registry, config);
        let mut wrapped = layer.layer(svc);

        wrapped.ready().await.unwrap();
        wrapped.call(Request::new(Body::empty())).await.unwrap();

        let guard = pn.lock().unwrap();
        let provider = guard.as_ref().unwrap().as_ref();
        assert!(
            provider.is_none(),
            "ProviderName should NOT be set when no provider matches"
        );
    }

    // ── Tests: layer/service debug impls ──────────────────────────────────────

    #[test]
    fn test_model_override_layer_debug() {
        let selector = static_selector("claude-sonnet-4-20250514");
        let registry = empty_registry();
        let config = empty_providers_config();
        let layer = ModelOverrideLayer::new(selector, registry, config);
        let s = format!("{layer:?}");
        assert!(s.contains("ModelOverrideLayer"));
    }
}
