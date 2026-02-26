//! Upstream auth-injection Tower layer.
//!
//! This layer runs **after** the proxy handler has determined which provider
//! handles a request (via the model-override layer).  Its job is to select a
//! key from the appropriate [`KeyPool`] and record that selection so the
//! provider's `send()` method can inject the credential into the upstream
//! request.
//!
//! # How key selection is communicated
//!
//! Tower middleware wraps the entire service call stack; it cannot intercept
//! the individual `reqwest` calls made inside a provider.  The practical
//! approach is to store the selected key's ID in the **request extensions**
//! as a [`SelectedKeyId`].  Provider implementations call
//! [`resolve_key`] to retrieve the full [`PooledKey`] from the pool by name
//! and ID.
//!
//! # Ordering
//!
//! This layer is placed **after** [`super::auth_layer::AuthLayer`] so that
//! [`crate::auth::validator::ValidatedClient`] is already present in extensions
//! when selection runs (useful for sticky/per-user selector strategies).  It
//! reads [`super::model_override::ProviderName`] — set by the model-override
//! layer — to pick the correct pool.
//!
//! If no [`ProviderName`] extension is present (e.g. for non-proxy routes like
//! `/health`) the layer passes the request through unchanged.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::Body;
use http::{Request, Response};
use tower::{Layer, Service};

use crate::key_pool::{KeyPool, PooledKey};
use crate::middleware::model_override::ProviderName;
use switchboard_common::types::RequestContext;

// ── Extension type ────────────────────────────────────────────────────────────

/// Extension inserted by [`AuthInjectLayer`] that records which key was chosen
/// for the current request.
///
/// Handlers and provider `send()` implementations retrieve the full
/// [`PooledKey`] via [`resolve_key`].
#[derive(Clone, Debug)]
pub struct SelectedKeyId(pub String);

// ── Layer ─────────────────────────────────────────────────────────────────────

/// Tower [`Layer`] that selects a key from the matching [`KeyPool`] and inserts
/// [`SelectedKeyId`] into request extensions.
#[derive(Clone)]
pub struct AuthInjectLayer {
    key_pools: Arc<HashMap<String, Arc<KeyPool>>>,
}

impl std::fmt::Debug for AuthInjectLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthInjectLayer")
            .field("pool_count", &self.key_pools.len())
            .finish()
    }
}

impl AuthInjectLayer {
    /// Create a new layer backed by the given key-pool map.
    ///
    /// `key_pools` maps short provider names (e.g. `"anthropic"`) to their
    /// runtime [`KeyPool`].
    pub fn new(key_pools: Arc<HashMap<String, Arc<KeyPool>>>) -> Self {
        Self { key_pools }
    }
}

impl<S> Layer<S> for AuthInjectLayer {
    type Service = AuthInjectService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthInjectService {
            inner,
            key_pools: Arc::clone(&self.key_pools),
        }
    }
}

// ── Service ───────────────────────────────────────────────────────────────────

/// Tower [`Service`] produced by [`AuthInjectLayer`].
#[derive(Clone)]
pub struct AuthInjectService<S> {
    inner: S,
    key_pools: Arc<HashMap<String, Arc<KeyPool>>>,
}

impl<S> std::fmt::Debug for AuthInjectService<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthInjectService").finish_non_exhaustive()
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

impl<S> Service<Request<Body>> for AuthInjectService<S>
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
        let key_pools = Arc::clone(&self.key_pools);
        let mut inner = self.inner.clone();

        Box::pin(async move {
            // Only act when a ProviderName extension was set by the
            // model-override layer.  Non-proxy routes won't have one.
            let provider_name = match req.extensions().get::<ProviderName>() {
                Some(p) => p.0.clone(),
                None => {
                    tracing::trace!(
                        "auth_inject: no ProviderName extension, skipping key selection"
                    );
                    return inner.call(req).await;
                }
            };

            let pool = match key_pools.get(&provider_name) {
                Some(p) => Arc::clone(p),
                None => {
                    tracing::debug!(
                        provider = %provider_name,
                        "auth_inject: no key pool found for provider, passing through"
                    );
                    return inner.call(req).await;
                }
            };

            // Build a minimal RequestContext for the selector.
            // User-identity fields are not yet populated at this layer;
            // they are available to selectors that need them via extensions
            // in later phases.
            let ctx = RequestContext::default();

            match pool.select(&ctx) {
                Some(key) => {
                    tracing::debug!(
                        provider = %provider_name,
                        key_id = %key.id,
                        "auth_inject: selected key for provider"
                    );
                    let mut req = req;
                    req.extensions_mut().insert(SelectedKeyId(key.id.clone()));
                    inner.call(req).await
                }
                None => {
                    // No eligible key — pass through without setting the
                    // extension.  The handler will return 503 as before.
                    tracing::debug!(
                        provider = %provider_name,
                        "auth_inject: no eligible key available, passing through"
                    );
                    inner.call(req).await
                }
            }
        })
    }
}

// ── Helper ────────────────────────────────────────────────────────────────────

/// Look up a [`PooledKey`] by provider name and key ID.
///
/// Used by provider `send()` implementations that want to retrieve the full
/// credential from the extension set by [`AuthInjectLayer`].
///
/// Returns `None` when either the provider pool or the key ID is not found.
pub fn resolve_key<'a>(
    pools: &'a HashMap<String, Arc<KeyPool>>,
    provider: &str,
    key_id: &str,
) -> Option<&'a PooledKey> {
    pools.get(provider)?.keys.iter().find(|k| k.id == key_id)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::task::Poll;

    use axum::body::Body;
    use http::{HeaderName, HeaderValue, Request, Response, StatusCode};
    use tower::{Layer, Service, ServiceExt};

    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::health::KeyStatus;
    use crate::key_pool::selector::WeightedRandomSelector;
    use crate::key_pool::{KeyPool, PooledKey};
    use crate::middleware::model_override::ProviderName;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_key(id: &str) -> PooledKey {
        PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer sk-test"),
                expires_at: None,
            },
            1.0,
        )
    }

    fn make_pool(keys: Vec<PooledKey>) -> Arc<KeyPool> {
        Arc::new(KeyPool::new(keys, Box::new(WeightedRandomSelector)))
    }

    fn make_pools(entries: Vec<(&str, Arc<KeyPool>)>) -> Arc<HashMap<String, Arc<KeyPool>>> {
        Arc::new(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    // A simple inner service that captures request extensions.
    #[derive(Clone)]
    struct CaptureSvc(Arc<std::sync::Mutex<Option<Option<SelectedKeyId>>>>);

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
            let key_id = req.extensions().get::<SelectedKeyId>().cloned();
            *self.0.lock().unwrap() = Some(key_id);
            std::future::ready(Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::empty())
                .unwrap()))
        }
    }

    // ── Unit tests: SelectedKeyId derive impls ────────────────────────────────

    #[test]
    fn test_selected_key_id_clone_and_debug() {
        let kid = SelectedKeyId("my-key-id".to_string());
        let cloned = kid.clone();
        assert_eq!(cloned.0, "my-key-id");

        let debug_str = format!("{kid:?}");
        assert!(debug_str.contains("my-key-id"));
    }

    // ── Unit tests: resolve_key helper ────────────────────────────────────────

    #[test]
    fn test_resolve_key_finds_by_provider_and_id() {
        let pool = make_pool(vec![make_key("k1"), make_key("k2")]);
        let pools = make_pools(vec![("anthropic", pool)]);

        let found = resolve_key(&pools, "anthropic", "k2");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "k2");
    }

    #[test]
    fn test_resolve_key_missing_provider_returns_none() {
        let pool = make_pool(vec![make_key("k1")]);
        let pools = make_pools(vec![("anthropic", pool)]);

        let found = resolve_key(&pools, "openai", "k1");
        assert!(found.is_none());
    }

    #[test]
    fn test_resolve_key_missing_key_id_returns_none() {
        let pool = make_pool(vec![make_key("k1")]);
        let pools = make_pools(vec![("anthropic", pool)]);

        let found = resolve_key(&pools, "anthropic", "no-such-key");
        assert!(found.is_none());
    }

    // ── Integration-style tests: layer behaviour ──────────────────────────────

    #[tokio::test]
    async fn test_key_id_inserted_when_provider_name_present() {
        let pool = make_pool(vec![make_key("k1")]);
        let pools = make_pools(vec![("anthropic", pool)]);

        let captured = Arc::new(std::sync::Mutex::new(None));
        let layer = AuthInjectLayer::new(pools);
        let mut svc = layer.layer(CaptureSvc(captured.clone()));

        let mut req = Request::new(Body::empty());
        req.extensions_mut()
            .insert(ProviderName("anthropic".into()));

        svc.ready().await.unwrap();
        svc.call(req).await.unwrap();

        let guard = captured.lock().unwrap();
        let key_id = guard.as_ref().unwrap().as_ref();
        assert!(key_id.is_some(), "SelectedKeyId should be set");
        assert_eq!(key_id.unwrap().0, "k1");
    }

    #[tokio::test]
    async fn test_no_provider_name_skips_injection() {
        let pool = make_pool(vec![make_key("k1")]);
        let pools = make_pools(vec![("anthropic", pool)]);

        let captured = Arc::new(std::sync::Mutex::new(None));
        let layer = AuthInjectLayer::new(pools);
        let mut svc = layer.layer(CaptureSvc(captured.clone()));

        // No ProviderName extension in this request.
        svc.ready().await.unwrap();
        svc.call(Request::new(Body::empty())).await.unwrap();

        let guard = captured.lock().unwrap();
        // CaptureSvc sets Some(None) when no SelectedKeyId is present.
        assert!(
            guard.as_ref().unwrap().is_none(),
            "SelectedKeyId should not be set when no ProviderName"
        );
    }

    #[tokio::test]
    async fn test_no_pool_for_provider_passes_through_without_extension() {
        // Empty pool map — provider name is present but no matching pool.
        let pools = make_pools(vec![]);

        let captured = Arc::new(std::sync::Mutex::new(None));
        let layer = AuthInjectLayer::new(pools);
        let mut svc = layer.layer(CaptureSvc(captured.clone()));

        let mut req = Request::new(Body::empty());
        req.extensions_mut().insert(ProviderName("openai".into()));

        svc.ready().await.unwrap();
        svc.call(req).await.unwrap();

        let guard = captured.lock().unwrap();
        assert!(
            guard.as_ref().unwrap().is_none(),
            "SelectedKeyId should not be set when no matching pool"
        );
    }

    #[tokio::test]
    async fn test_no_eligible_key_passes_through_without_extension() {
        let mut key = make_key("k1");
        key.health.status = KeyStatus::Disabled;
        let pool = make_pool(vec![key]);
        let pools = make_pools(vec![("anthropic", pool)]);

        let captured = Arc::new(std::sync::Mutex::new(None));
        let layer = AuthInjectLayer::new(pools);
        let mut svc = layer.layer(CaptureSvc(captured.clone()));

        let mut req = Request::new(Body::empty());
        req.extensions_mut()
            .insert(ProviderName("anthropic".into()));

        svc.ready().await.unwrap();
        svc.call(req).await.unwrap();

        let guard = captured.lock().unwrap();
        assert!(
            guard.as_ref().unwrap().is_none(),
            "SelectedKeyId should not be set when all keys are disabled"
        );
    }

    #[test]
    fn test_auth_inject_layer_debug() {
        let pools = make_pools(vec![]);
        let layer = AuthInjectLayer::new(pools);
        let s = format!("{layer:?}");
        assert!(s.contains("AuthInjectLayer"));
    }
}
