//! Request-ID Tower layer.
//!
//! Generates a fresh `X-Switchboard-Request-Id` UUID on every incoming
//! request and inserts it into:
//!
//! 1. The request extensions (as a [`RequestId`] newtype), so downstream
//!    handlers can retrieve it without re-parsing.
//! 2. The response extensions (for observability).
//!
//! If the incoming request already carries a `X-Switchboard-Request-Id`
//! header the existing value is reused verbatim to preserve end-to-end
//! correlation.

use std::task::{Context, Poll};

use http::{Request, Response};
use tower::{Layer, Service};
use uuid::Uuid;

use switchboard_common::protocol::HEADER_REQUEST_ID;

// ── RequestId newtype ─────────────────────────────────────────────────────────

/// Newtype wrapper around a request-id string, stored in request extensions.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

impl RequestId {
    /// The string value of the request ID.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── Layer ─────────────────────────────────────────────────────────────────────

/// Tower [`Layer`] that wraps a service with [`RequestIdService`].
#[derive(Debug, Clone, Default)]
pub struct RequestIdLayer;

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdService { inner }
    }
}

// ── Service ───────────────────────────────────────────────────────────────────

/// Tower [`Service`] that injects a `RequestId` into request extensions.
#[derive(Debug, Clone)]
pub struct RequestIdService<S> {
    inner: S,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for RequestIdService<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request<ReqBody>) -> Self::Future {
        // Re-use an existing request-id header if present; otherwise generate.
        let id_str = req
            .headers()
            .get(HEADER_REQUEST_ID)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let request_id = RequestId(id_str);
        tracing::debug!(request_id = %request_id, "assigned request ID");
        req.extensions_mut().insert(request_id);

        self.inner.call(req)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::task::Poll;

    use http::{Request, Response};
    use tower::{Layer, Service, ServiceExt};

    use super::*;

    // A generic Clone-friendly inner service that checks extensions.
    #[derive(Clone)]
    struct CheckIdSvc {
        expected_id: Option<String>,
        captured: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl Service<Request<()>> for CheckIdSvc {
        type Response = Response<()>;
        type Error = Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(
            &mut self,
            _cx: &mut std::task::Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, req: Request<()>) -> Self::Future {
            let id = req.extensions().get::<RequestId>().cloned();
            if let Some(ref id) = id {
                self.captured.lock().unwrap().push(id.0.clone());
                if let Some(ref expected) = self.expected_id {
                    assert_eq!(id.as_str(), expected, "request id mismatch");
                }
            }
            std::future::ready(Ok(Response::new(())))
        }
    }

    #[tokio::test]
    async fn test_request_id_injected_into_extensions() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let layer = RequestIdLayer;
        let mut svc = layer.layer(CheckIdSvc {
            expected_id: None,
            captured: captured.clone(),
        });

        svc.ready().await.unwrap();
        svc.call(Request::new(())).await.unwrap();

        let ids = captured.lock().unwrap();
        assert_eq!(ids.len(), 1);
        assert!(!ids[0].is_empty());
        // Should parse as a valid UUID.
        Uuid::parse_str(&ids[0]).expect("request id should be a valid UUID");
    }

    #[tokio::test]
    async fn test_existing_request_id_header_reused() {
        let existing_id = "550e8400-e29b-41d4-a716-446655440000";
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let layer = RequestIdLayer;
        let mut svc = layer.layer(CheckIdSvc {
            expected_id: Some(existing_id.to_owned()),
            captured: captured.clone(),
        });

        let mut req = Request::new(());
        req.headers_mut()
            .insert(HEADER_REQUEST_ID, existing_id.parse().unwrap());

        svc.ready().await.unwrap();
        svc.call(req).await.unwrap();

        let ids = captured.lock().unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], existing_id);
    }

    #[tokio::test]
    async fn test_each_request_gets_unique_id() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let layer = RequestIdLayer;
        let mut svc = layer.layer(CheckIdSvc {
            expected_id: None,
            captured: captured.clone(),
        });

        for _ in 0..3 {
            svc.ready().await.unwrap();
            svc.call(Request::new(())).await.unwrap();
        }

        let ids = captured.lock().unwrap();
        assert_eq!(ids.len(), 3);
        // All three should be distinct.
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_request_id_display() {
        let id = RequestId("test-123".into());
        assert_eq!(id.to_string(), "test-123");
        assert_eq!(id.as_str(), "test-123");
    }
}
