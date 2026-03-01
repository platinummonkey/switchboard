//! Upstream auth-injection Tower layer.

use crate::key_pool::{KeyPool, PooledKey};
use crate::middleware::model_override::ProviderName;
use axum::body::Body;
use http::{Request, Response};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};
use switchboard_common::types::RequestContext;
use tower::{Layer, Service};

#[derive(Clone, Debug)]
pub struct SelectedKeyId(pub String);

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
                    tracing::debug!(provider = %provider_name, "auth_inject: no key pool found");
                    return inner.call(req).await;
                }
            };
            let ctx = RequestContext::default();
            match pool.select(&ctx) {
                Some(key_arc) => {
                    let key_id = key_arc.read().unwrap().id.clone();
                    tracing::debug!(provider = %provider_name, key_id = %key_id, "auth_inject: selected key");
                    let mut req = req;
                    req.extensions_mut().insert(SelectedKeyId(key_id));
                    inner.call(req).await
                }
                None => {
                    tracing::debug!(provider = %provider_name, "auth_inject: no eligible key");
                    inner.call(req).await
                }
            }
        })
    }
}

pub fn resolve_key(
    pools: &HashMap<String, Arc<KeyPool>>,
    provider: &str,
    key_id: &str,
) -> Option<Arc<RwLock<PooledKey>>> {
    pools.get(provider)?.get_key(key_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::health::KeyStatus;
    use crate::key_pool::selector::WeightedRandomSelector;
    use crate::key_pool::{KeyPool, PooledKey};
    use crate::middleware::model_override::ProviderName;
    use axum::body::Body;
    use http::{HeaderName, HeaderValue, Request, Response, StatusCode};
    use std::collections::HashMap;
    use std::convert::Infallible;
    use std::sync::{Arc, RwLock};
    use std::task::Poll;
    use tower::{Layer, Service, ServiceExt};
    fn mk(id: &str) -> Arc<RwLock<PooledKey>> {
        Arc::new(RwLock::new(PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer sk-test"),
                expires_at: None,
            },
            1.0,
        )))
    }
    fn mpool(keys: Vec<Arc<RwLock<PooledKey>>>) -> Arc<KeyPool> {
        Arc::new(KeyPool::new(keys, Box::new(WeightedRandomSelector)))
    }
    fn mpools(entries: Vec<(&str, Arc<KeyPool>)>) -> Arc<HashMap<String, Arc<KeyPool>>> {
        Arc::new(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }
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
            let kid = req.extensions().get::<SelectedKeyId>().cloned();
            *self.0.lock().unwrap() = Some(kid);
            std::future::ready(Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::empty())
                .unwrap()))
        }
    }
    #[test]
    fn test_selected_key_id_clone_and_debug() {
        let kid = SelectedKeyId("my-key-id".to_string());
        let cloned = kid.clone();
        assert_eq!(cloned.0, "my-key-id");
        assert!(format!("{kid:?}").contains("my-key-id"));
    }
    #[test]
    fn test_resolve_key_finds_by_provider_and_id() {
        let pool = mpool(vec![mk("k1"), mk("k2")]);
        let pools = mpools(vec![("anthropic", pool)]);
        let found = resolve_key(&pools, "anthropic", "k2");
        assert!(found.is_some());
        assert_eq!(found.unwrap().read().unwrap().id, "k2");
    }
    #[test]
    fn test_resolve_key_missing_provider_returns_none() {
        let pool = mpool(vec![mk("k1")]);
        let pools = mpools(vec![("anthropic", pool)]);
        assert!(resolve_key(&pools, "openai", "k1").is_none());
    }
    #[test]
    fn test_resolve_key_missing_key_id_returns_none() {
        let pool = mpool(vec![mk("k1")]);
        let pools = mpools(vec![("anthropic", pool)]);
        assert!(resolve_key(&pools, "anthropic", "no-such-key").is_none());
    }
    #[tokio::test]
    async fn test_key_id_inserted_when_provider_name_present() {
        let pool = mpool(vec![mk("k1")]);
        let pools = mpools(vec![("anthropic", pool)]);
        let captured = Arc::new(std::sync::Mutex::new(None));
        let layer = AuthInjectLayer::new(pools);
        let mut svc = layer.layer(CaptureSvc(captured.clone()));
        let mut req = Request::new(Body::empty());
        req.extensions_mut()
            .insert(ProviderName("anthropic".into()));
        svc.ready().await.unwrap();
        svc.call(req).await.unwrap();
        let guard = captured.lock().unwrap();
        assert!(guard.as_ref().unwrap().as_ref().is_some());
        assert_eq!(guard.as_ref().unwrap().as_ref().unwrap().0, "k1");
    }
    #[tokio::test]
    async fn test_no_provider_name_skips_injection() {
        let pool = mpool(vec![mk("k1")]);
        let pools = mpools(vec![("anthropic", pool)]);
        let captured = Arc::new(std::sync::Mutex::new(None));
        let mut svc = AuthInjectLayer::new(pools).layer(CaptureSvc(captured.clone()));
        svc.ready().await.unwrap();
        svc.call(Request::new(Body::empty())).await.unwrap();
        assert!(captured.lock().unwrap().as_ref().unwrap().is_none());
    }
    #[tokio::test]
    async fn test_no_pool_for_provider_passes_through_without_extension() {
        let captured = Arc::new(std::sync::Mutex::new(None));
        let mut svc = AuthInjectLayer::new(mpools(vec![])).layer(CaptureSvc(captured.clone()));
        let mut req = Request::new(Body::empty());
        req.extensions_mut().insert(ProviderName("openai".into()));
        svc.ready().await.unwrap();
        svc.call(req).await.unwrap();
        assert!(captured.lock().unwrap().as_ref().unwrap().is_none());
    }
    #[tokio::test]
    async fn test_no_eligible_key_passes_through_without_extension() {
        let key = mk("k1");
        key.write().unwrap().health.status = KeyStatus::Disabled;
        let captured = Arc::new(std::sync::Mutex::new(None));
        let mut svc = AuthInjectLayer::new(mpools(vec![("anthropic", mpool(vec![key]))]))
            .layer(CaptureSvc(captured.clone()));
        let mut req = Request::new(Body::empty());
        req.extensions_mut()
            .insert(ProviderName("anthropic".into()));
        svc.ready().await.unwrap();
        svc.call(req).await.unwrap();
        assert!(captured.lock().unwrap().as_ref().unwrap().is_none());
    }
    #[test]
    fn test_auth_inject_layer_debug() {
        let pools = mpools(vec![]);
        let layer = AuthInjectLayer::new(pools);
        assert!(format!("{layer:?}").contains("AuthInjectLayer"));
    }
}
