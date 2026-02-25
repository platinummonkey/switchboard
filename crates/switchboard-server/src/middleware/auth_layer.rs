//! Auth Tower layer.
//!
//! Extracts the `Authorization` header from incoming requests, delegates
//! validation to an [`AuthRegistry`], and either:
//!
//! - Inserts the resulting [`ValidatedClient`] into request extensions
//!   so downstream handlers can access caller identity without re-validating.
//! - Returns HTTP 401 Unauthorized if all validators reject the request.
//!
//! Requests without an `Authorization` header are rejected with 401.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::response::IntoResponse;
use http::{Request, Response, StatusCode};
use tower::{Layer, Service};

use crate::auth::registry::AuthRegistry;
use crate::auth::validator::ValidatedClient;

// ── Layer ─────────────────────────────────────────────────────────────────────

/// Tower [`Layer`] that wraps services with [`AuthService`].
#[derive(Clone)]
pub struct AuthLayer {
    registry: Arc<AuthRegistry>,
}

impl std::fmt::Debug for AuthLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthLayer").finish_non_exhaustive()
    }
}

impl AuthLayer {
    /// Create a new layer backed by the given registry.
    pub fn new(registry: Arc<AuthRegistry>) -> Self {
        Self { registry }
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = AuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthService {
            inner,
            registry: Arc::clone(&self.registry),
        }
    }
}

// ── Service ───────────────────────────────────────────────────────────────────

/// Tower [`Service`] that validates the `Authorization` header and injects
/// [`ValidatedClient`] into request extensions.
#[derive(Clone)]
pub struct AuthService<S> {
    inner: S,
    registry: Arc<AuthRegistry>,
}

impl<S> std::fmt::Debug for AuthService<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthService").finish_non_exhaustive()
    }
}

/// A boxed, pinned future for the response.
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

impl<S> Service<Request<Body>> for AuthService<S>
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
        let registry = Arc::clone(&self.registry);
        let mut inner = self.inner.clone();

        Box::pin(async move {
            // Extract the Authorization header value.
            let auth_header = match req.headers().get(http::header::AUTHORIZATION) {
                Some(v) => match v.to_str() {
                    Ok(s) => s.to_owned(),
                    Err(_) => {
                        tracing::debug!("Authorization header contains non-UTF8 bytes");
                        return Ok(unauthorized_response("invalid authorization header"));
                    }
                },
                None => {
                    tracing::debug!("request missing Authorization header");
                    return Ok(unauthorized_response("missing authorization header"));
                }
            };

            // Delegate to the registry.
            match registry.validate(&auth_header).await {
                Ok(client) => {
                    tracing::debug!(
                        user_id = ?client.user_id,
                        "request authenticated"
                    );
                    let mut req = req;
                    req.extensions_mut().insert(client);
                    inner.call(req).await
                }
                Err(e) => {
                    tracing::debug!(error = %e, "auth rejected");
                    Ok(unauthorized_response(&e.to_string()))
                }
            }
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unauthorized_response(reason: &str) -> Response<Body> {
    let body = serde_json::json!({
        "error": "unauthorized",
        "message": reason,
    })
    .to_string();

    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("static response is always valid")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;

    use async_trait::async_trait;
    use axum::body::Body;
    use http::{Request, Response, StatusCode};
    use tower::{Layer, Service, ServiceExt};

    use super::*;
    use crate::auth::registry::AuthRegistry;
    use crate::auth::validator::{AuthError, ClientAuthValidator, ValidatedClient};

    struct AllowAll;

    #[async_trait]
    impl ClientAuthValidator for AllowAll {
        fn name(&self) -> &str {
            "allow-all"
        }

        async fn validate(&self, _h: &str) -> Result<ValidatedClient, AuthError> {
            Ok(ValidatedClient::from_static_key())
        }
    }

    struct DenyAll;

    #[async_trait]
    impl ClientAuthValidator for DenyAll {
        fn name(&self) -> &str {
            "deny-all"
        }

        async fn validate(&self, _h: &str) -> Result<ValidatedClient, AuthError> {
            Err(AuthError::Unauthorized("denied".into()))
        }
    }

    fn allow_registry() -> Arc<AuthRegistry> {
        Arc::new(AuthRegistry::builder().add(AllowAll).build())
    }

    fn deny_registry() -> Arc<AuthRegistry> {
        Arc::new(AuthRegistry::builder().add(DenyAll).build())
    }

    fn empty_registry() -> Arc<AuthRegistry> {
        Arc::new(AuthRegistry::new(vec![]))
    }

    // A Clone-friendly inner service for tests.
    #[derive(Clone)]
    struct OkSvc;

    impl Service<Request<Body>> for OkSvc {
        type Response = Response<Body>;
        type Error = Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: Request<Body>) -> Self::Future {
            std::future::ready(Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::empty())
                .unwrap()))
        }
    }

    #[tokio::test]
    async fn test_valid_auth_passes_through() {
        let layer = AuthLayer::new(allow_registry());
        let mut svc = layer.layer(OkSvc);

        let mut req = Request::new(Body::empty());
        req.headers_mut().insert(
            http::header::AUTHORIZATION,
            "Bearer sk-valid".parse().unwrap(),
        );

        svc.ready().await.unwrap();
        let resp = svc.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_missing_auth_header_returns_401() {
        let layer = AuthLayer::new(allow_registry());
        let mut svc = layer.layer(OkSvc);

        svc.ready().await.unwrap();
        let resp = svc.call(Request::new(Body::empty())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_denied_auth_returns_401() {
        let layer = AuthLayer::new(deny_registry());
        let mut svc = layer.layer(OkSvc);

        let mut req = Request::new(Body::empty());
        req.headers_mut().insert(
            http::header::AUTHORIZATION,
            "Bearer sk-bad".parse().unwrap(),
        );

        svc.ready().await.unwrap();
        let resp = svc.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_empty_registry_returns_401() {
        let layer = AuthLayer::new(empty_registry());
        let mut svc = layer.layer(OkSvc);

        let mut req = Request::new(Body::empty());
        req.headers_mut().insert(
            http::header::AUTHORIZATION,
            "Bearer sk-any".parse().unwrap(),
        );

        svc.ready().await.unwrap();
        let resp = svc.call(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_validated_client_inserted_in_extensions() {
        let client_captured = std::sync::Arc::new(std::sync::Mutex::new(None::<ValidatedClient>));
        let cc = client_captured.clone();

        #[derive(Clone)]
        struct CaptureSvc(std::sync::Arc<std::sync::Mutex<Option<ValidatedClient>>>);

        impl Service<Request<Body>> for CaptureSvc {
            type Response = Response<Body>;
            type Error = Infallible;
            type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, req: Request<Body>) -> Self::Future {
                let client = req.extensions().get::<ValidatedClient>().cloned();
                *self.0.lock().unwrap() = client;
                std::future::ready(Ok(Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::empty())
                    .unwrap()))
            }
        }

        let layer = AuthLayer::new(allow_registry());
        let mut svc = layer.layer(CaptureSvc(cc));

        let mut req = Request::new(Body::empty());
        req.headers_mut()
            .insert(http::header::AUTHORIZATION, "Bearer sk-ok".parse().unwrap());

        svc.ready().await.unwrap();
        svc.call(req).await.unwrap();

        let captured = client_captured.lock().unwrap();
        assert!(
            captured.is_some(),
            "ValidatedClient should be in extensions"
        );
    }
}
