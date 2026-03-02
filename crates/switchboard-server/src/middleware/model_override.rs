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
//! The layer buffers the request body (up to 1 MiB) to extract the `"model"`
//! field from the JSON payload.  This enables `mapping` mode to fire on the
//! body model when no `x-switchboard-model` header is present.  The buffered
//! bytes are re-attached to the request so the downstream handler can read the
//! body as usual.
//!
//! Priority order:
//! 1. `x-switchboard-model` header — explicit client override (no body parse).
//! 2. `selector.select(&ctx)` where `ctx.model` comes from the JSON body and
//!    `ctx.user_id` is resolved from three sources (first wins):
//!    - `x-switchboard-user` header — explicit client identity override.
//!    - [`crate::auth::validator::ValidatedClient`]`.user_id` from extensions,
//!      set by [`super::auth_layer::AuthLayer`] when a static API key is
//!      mapped via `identity.api_key_mappings`.  Because `AuthLayer` runs
//!      before `ModelOverrideLayer`, this extension is always available.
//!    - [`crate::identity::MtlsClientCn`] extension — set by the TLS accept
//!      loop when a client certificate is present. Allows per-user model
//!      overrides keyed on the mTLS Common Name.
//!
//!    This triple-source identity enables `mapping` mode and per-user `dynamic`
//!    overrides for api-key-authenticated and mTLS-authenticated users.
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
use bytes::Bytes;
use http::{Request, Response};
use tower::{Layer, Service};

use switchboard_common::protocol::{HEADER_MODEL, HEADER_USER};
use switchboard_common::types::RequestContext;

use crate::auth::validator::ValidatedClient;
use crate::config::provider::ProvidersConfig;
use crate::identity::MtlsClientCn;
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
            // Split the request into parts so we can take ownership of the
            // body while still reading the headers.
            let (parts, body) = req.into_parts();

            let (resolved_model, reason, body_bytes) = if let Some(header_model) = parts
                .headers
                .get(HEADER_MODEL)
                .and_then(|v| v.to_str().ok())
            {
                // Header takes precedence — skip body parsing entirely.
                let header_model = header_model.to_owned();
                tracing::debug!(
                    model = %header_model,
                    "model_override: model from x-switchboard-model header"
                );
                let bytes = axum::body::to_bytes(body, usize::MAX)
                    .await
                    .unwrap_or_default();
                (header_model, SelectionReason::HeaderOverride, bytes)
            } else {
                // Buffer the body so we can read the JSON "model" field.
                // Cap at 1 MiB to guard against enormous payloads; if the
                // read fails we fall through with no body model (graceful
                // degradation: selector still applies static/fallback policy).
                const MAX_PEEK_BYTES: usize = 1024 * 1024;
                let body_bytes: Bytes = match axum::body::to_bytes(body, MAX_PEEK_BYTES).await {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::debug!(
                            error = %e,
                            "model_override: failed to buffer body, using empty context"
                        );
                        Bytes::new()
                    }
                };

                // Parse the JSON "model" field from the buffered bytes.
                let body_model: Option<String> =
                    serde_json::from_slice::<serde_json::Value>(&body_bytes)
                        .ok()
                        .and_then(|v| v["model"].as_str().map(|s| s.to_string()));

                // Resolve user identity for per-user model overrides.
                //
                // Priority order:
                //   1. `x-switchboard-user` header — explicit client override.
                //   2. `ValidatedClient.user_id` — set by AuthLayer when a
                //      static API key is mapped to a user identity via
                //      `identity.api_key_mappings` in the server config.
                //      AuthLayer runs before ModelOverrideLayer, so the
                //      extension is already populated at this point.
                //   3. `MtlsClientCn` extension — set by the TLS accept loop
                //      when a client certificate is present. Enables per-user
                //      model overrides keyed on the mTLS Common Name.
                //
                // This triple-source identity enables `mapping` mode and
                // per-user `dynamic` overrides for api-key-authenticated users
                // and mTLS-authenticated users without requiring an explicit
                // `x-switchboard-user` header.
                let user_id: Option<String> = parts
                    .headers
                    .get(HEADER_USER)
                    .and_then(|v| v.to_str().ok())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        parts
                            .extensions
                            .get::<ValidatedClient>()
                            .and_then(|c| c.user_id.clone())
                    })
                    .or_else(|| {
                        parts
                            .extensions
                            .get::<MtlsClientCn>()
                            .map(|cn| cn.0.clone())
                    });

                // Build a RequestContext enriched with what we know at this
                // point (body model + user_id from header).
                let mut ctx = RequestContext::new();
                if let Some(ref m) = body_model {
                    ctx.model = Some(m.clone());
                }
                if let Some(ref u) = user_id {
                    ctx.user_id = Some(u.clone());
                }

                let (model, reason) = selector.select(&ctx);
                tracing::debug!(
                    model = %model,
                    reason = %reason,
                    body_model = ?body_model,
                    user_id = ?user_id,
                    "model_override: model from selector policy"
                );

                (model, reason, body_bytes)
            };

            // Step 2: reconstruct the request from parts + buffered body bytes
            // so the downstream handler can read the body as usual.
            let mut req = Request::from_parts(parts, Body::from(body_bytes));

            // Step 3: resolve the provider for the model.
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
    use serde_json::json;
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

    fn mapping_selector(mappings: HashMap<String, String>, fallback: &str) -> Arc<ModelSelector> {
        Arc::new(ModelSelector::new(ModelSelectionConfig {
            mode: "mapping".into(),
            mappings,
            fallback: Some(fallback.to_string()),
            ..ModelSelectionConfig::default()
        }))
    }

    fn dynamic_selector_with_overrides(
        overrides: HashMap<String, String>,
        fallback: &str,
    ) -> Arc<ModelSelector> {
        Arc::new(ModelSelector::new(ModelSelectionConfig {
            mode: "dynamic".into(),
            overrides,
            fallback: Some(fallback.to_string()),
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

    // ── Tests: body-peeking for mapping mode ──────────────────────────────────

    #[tokio::test]
    async fn test_mapping_mode_fires_on_body_model() {
        // Configure mapping: gpt-4 → gpt-4o
        let mut mappings = HashMap::new();
        mappings.insert("gpt-4".to_string(), "gpt-4o".to_string());
        let selector = mapping_selector(mappings, "gpt-4o");
        let registry = registry_with_openai(vec!["gpt-4".to_string(), "gpt-4o".to_string()]);
        let config =
            providers_config_with("openai", vec!["gpt-4".to_string(), "gpt-4o".to_string()]);

        let (svc, rm, _pn) = CaptureSvc::new();
        let layer = ModelOverrideLayer::new(selector, registry, config);
        let mut wrapped = layer.layer(svc);

        // Send a request with "model": "gpt-4" in the body; no header.
        let body = serde_json::to_vec(&json!({"model": "gpt-4", "messages": []})).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();

        wrapped.ready().await.unwrap();
        wrapped.call(req).await.unwrap();

        let guard = rm.lock().unwrap();
        let resolved = guard.as_ref().unwrap().as_ref();
        assert!(resolved.is_some(), "ResolvedModel should be set");
        // Mapping fires: gpt-4 → gpt-4o
        assert_eq!(
            resolved.unwrap().0,
            "gpt-4o",
            "mapping mode must fire on body model"
        );
    }

    #[tokio::test]
    async fn test_body_peeking_per_user_override_fires() {
        // Configure dynamic mode with a per-user override for "alice" → "gpt-3.5-turbo"
        let mut overrides = HashMap::new();
        overrides.insert("alice".to_string(), "gpt-3.5-turbo".to_string());
        let selector = dynamic_selector_with_overrides(overrides, "gpt-4o");
        let registry =
            registry_with_openai(vec!["gpt-4o".to_string(), "gpt-3.5-turbo".to_string()]);
        let config = providers_config_with(
            "openai",
            vec!["gpt-4o".to_string(), "gpt-3.5-turbo".to_string()],
        );

        let (svc, rm, _pn) = CaptureSvc::new();
        let layer = ModelOverrideLayer::new(selector, registry, config);
        let mut wrapped = layer.layer(svc);

        // Send a request as "alice" with no x-switchboard-model header.
        let body = serde_json::to_vec(&json!({"model": "gpt-4o", "messages": []})).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(HEADER_USER, "alice")
            .body(Body::from(body))
            .unwrap();

        wrapped.ready().await.unwrap();
        wrapped.call(req).await.unwrap();

        let guard = rm.lock().unwrap();
        let resolved = guard.as_ref().unwrap().as_ref();
        assert!(resolved.is_some(), "ResolvedModel should be set");
        // Per-user override fires: alice → gpt-3.5-turbo
        assert_eq!(
            resolved.unwrap().0,
            "gpt-3.5-turbo",
            "per-user override must fire when x-switchboard-user header is present"
        );
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
