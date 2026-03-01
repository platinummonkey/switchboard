//! Token-bucket rate limiter Tower layer.
//!
//! Enforces per-user request-per-minute (RPM) and token-per-minute (TPM) limits
//! using a sliding window approach.  State is stored in a [`DashMap`] keyed by
//! user identifier.
//!
//! # Algorithm
//!
//! A simple fixed-window counter: each user has a [`BucketState`] that records
//! how many requests and tokens have been consumed in the **current minute
//! window**.  When a new request arrives:
//!
//! 1. If the current time is past `window_start + 60s`, reset the window.
//! 2. Check that `requests_this_window < effective_rpm` and
//!    `tokens_this_window + estimated_tokens <= effective_tpm`.
//! 3. If both limits are satisfied, increment and allow.  Otherwise return 429.
//!
//! Token estimation: when the exact token count is not known at request time
//! (before the LLM responds), a conservative estimate of 0 input tokens is
//! assumed; the caller updates the bucket post-response via
//! [`RateLimitHandle::record_tokens`] once the provider returns actual usage.
//!
//! Per-user overrides are read from the config supplied at construction time
//! and stored in `overrides`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use dashmap::DashMap;
use http::{Request, Response, StatusCode};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tower::{Layer, Service};

use serde::{Deserialize, Serialize};

use crate::auth::validator::ValidatedClient;

// ── RateLimitHandle ───────────────────────────────────────────────────────────

/// A handle to the live rate-limit state.
///
/// Cloning gives a reference to the same underlying state so changes made
/// through the handle are immediately visible to in-flight requests.
#[derive(Debug, Clone)]
pub struct RateLimitHandle {
    overrides: Arc<DashMap<String, RateLimitSettings>>,
    /// Shared reference to the full rate-limit state so `record_tokens` can
    /// update the per-user token bucket after a response is received.
    state: Arc<RateLimitState>,
}

impl RateLimitHandle {
    /// Set or update the rate limit for a specific user/team ID.
    pub fn set_override(&self, id: impl Into<String>, settings: RateLimitSettings) {
        self.overrides.insert(id.into(), settings);
    }

    /// Remove an override, reverting to the global default.
    pub fn remove_override(&self, id: &str) {
        self.overrides.remove(id);
    }

    /// Return a snapshot of all current overrides.
    pub fn snapshot(&self) -> Vec<(String, RateLimitSettings)> {
        self.overrides
            .iter()
            .map(|r| (r.key().clone(), *r.value()))
            .collect()
    }

    /// Record additional token usage for a user after the LLM responds.
    ///
    /// This updates the per-user token bucket so that TPM limits are enforced
    /// based on the actual token count returned by the upstream provider.
    /// The proxy handler calls this post-response with `input_tokens + output_tokens`.
    pub fn record_tokens(&self, user_id: &str, tokens: u32) {
        self.state.record_tokens(user_id, tokens);
    }
}

// ── Rate limit config snapshot ────────────────────────────────────────────────

/// A snapshot of rate limit settings for a single user/entity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RateLimitSettings {
    /// Max requests per minute.
    pub rpm: u32,
    /// Max tokens per minute.
    pub tpm: u32,
}

// ── Per-user bucket ────────────────────────────────────────────────────────────

#[derive(Debug)]
struct BucketState {
    /// Start of the current 60-second window.
    window_start: Instant,
    /// Number of requests consumed in the current window.
    requests: u32,
    /// Number of tokens consumed in the current window.
    tokens: u32,
}

impl BucketState {
    fn new() -> Self {
        Self {
            window_start: Instant::now(),
            requests: 0,
            tokens: 0,
        }
    }

    /// Advance window if stale; returns `true` if window was reset.
    fn maybe_reset(&mut self, now: Instant) -> bool {
        if now.duration_since(self.window_start) >= Duration::from_secs(60) {
            self.window_start = now;
            self.requests = 0;
            self.tokens = 0;
            true
        } else {
            false
        }
    }
}

// ── Layer ─────────────────────────────────────────────────────────────────────

/// Tower [`Layer`] that wraps services with [`RateLimitService`].
#[derive(Clone, Debug)]
pub struct RateLimitLayer {
    inner: Arc<RateLimitState>,
}

impl RateLimitLayer {
    /// Create a new layer and its associated [`RateLimitHandle`].
    ///
    /// Returns a `(layer, handle)` tuple so callers can update overrides at
    /// runtime without restarting. Both the layer and handle share the same
    /// underlying `Arc<RateLimitState>`, so token recordings and override
    /// changes via the handle are immediately visible to in-flight requests.
    ///
    /// - `default_rpm`: Requests per minute for users without an override.
    /// - `default_tpm`: Tokens per minute for users without an override.
    /// - `overrides`: Initial per-user settings keyed by user identifier string.
    pub fn new(
        default_rpm: u32,
        default_tpm: u32,
        overrides: impl IntoIterator<Item = (String, RateLimitSettings)>,
    ) -> (Self, RateLimitHandle) {
        let overrides_map: Arc<DashMap<String, RateLimitSettings>> =
            Arc::new(overrides.into_iter().collect());
        let state = Arc::new(RateLimitState {
            default_rpm,
            default_tpm,
            overrides: Arc::clone(&overrides_map),
            buckets: DashMap::new(),
        });
        let handle = RateLimitHandle {
            overrides: Arc::clone(&overrides_map),
            state: Arc::clone(&state),
        };
        let layer = Self { inner: state };
        (layer, handle)
    }
}

impl<S> Layer<S> for RateLimitLayer {
    type Service = RateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitService {
            inner,
            state: Arc::clone(&self.inner),
        }
    }
}

// ── Shared state ──────────────────────────────────────────────────────────────

#[derive(Debug)]
struct RateLimitState {
    default_rpm: u32,
    default_tpm: u32,
    /// Shared with [`RateLimitHandle`] via `Arc` so live updates are visible.
    overrides: Arc<DashMap<String, RateLimitSettings>>,
    buckets: DashMap<String, BucketState>,
}

impl RateLimitState {
    fn settings_for(&self, user_id: &str) -> RateLimitSettings {
        self.overrides
            .get(user_id)
            .map(|r| *r)
            .unwrap_or(RateLimitSettings {
                rpm: self.default_rpm,
                tpm: self.default_tpm,
            })
    }

    /// Check and record a request.  Returns `Ok(())` if allowed, `Err` with a
    /// human-readable reason if rate limited.
    fn check_and_record(
        &self,
        user_id: &str,
        estimated_tokens: u32,
    ) -> Result<(), RateLimitReason> {
        let settings = self.settings_for(user_id);
        let now = Instant::now();

        let mut bucket = self
            .buckets
            .entry(user_id.to_owned())
            .or_insert_with(BucketState::new);
        bucket.maybe_reset(now);

        if bucket.requests >= settings.rpm {
            return Err(RateLimitReason::Rpm {
                limit: settings.rpm,
                current: bucket.requests,
            });
        }

        if settings.tpm > 0 && bucket.tokens + estimated_tokens > settings.tpm {
            return Err(RateLimitReason::Tpm {
                limit: settings.tpm,
                current: bucket.tokens,
                requested: estimated_tokens,
            });
        }

        bucket.requests += 1;
        bucket.tokens += estimated_tokens;
        Ok(())
    }

    /// Update the token count for a user after the actual token usage is known.
    /// `additional_tokens` is added to the current window; called
    /// after the LLM responds with actual usage via [`RateLimitHandle::record_tokens`].
    pub fn record_tokens(&self, user_id: &str, additional_tokens: u32) {
        if let Some(mut bucket) = self.buckets.get_mut(user_id) {
            bucket.tokens = bucket.tokens.saturating_add(additional_tokens);
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
enum RateLimitReason {
    Rpm {
        limit: u32,
        current: u32,
    },
    Tpm {
        limit: u32,
        current: u32,
        requested: u32,
    },
    Anonymous,
}

impl std::fmt::Display for RateLimitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLimitReason::Rpm { limit, current } => write!(
                f,
                "rate limit exceeded: {current} requests in current window (limit: {limit} rpm)"
            ),
            RateLimitReason::Tpm {
                limit,
                current,
                requested,
            } => write!(
                f,
                "rate limit exceeded: {current} tokens used + {requested} requested exceeds {limit} tpm"
            ),
            RateLimitReason::Anonymous => {
                write!(f, "rate limit exceeded for anonymous requests")
            }
        }
    }
}

// ── Service ───────────────────────────────────────────────────────────────────

/// Tower [`Service`] that enforces per-user rate limits.
#[derive(Clone)]
pub struct RateLimitService<S> {
    inner: S,
    state: Arc<RateLimitState>,
}

impl<S> std::fmt::Debug for RateLimitService<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimitService").finish_non_exhaustive()
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

impl<S> Service<Request<Body>> for RateLimitService<S>
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
        let state = Arc::clone(&self.state);
        let mut inner = self.inner.clone();

        Box::pin(async move {
            // Determine the user identifier.
            // We look for a `ValidatedClient` in extensions (set by AuthLayer),
            // falling back to a synthetic "anonymous" key so rate limiting still
            // applies to unauthenticated requests (if auth is not required).
            let user_id = req
                .extensions()
                .get::<ValidatedClient>()
                .and_then(|c| c.user_id.clone())
                .unwrap_or_else(|| "anonymous".to_owned());

            // No pre-known token count at request time.
            match state.check_and_record(&user_id, 0) {
                Ok(()) => {
                    tracing::debug!(user_id = %user_id, "rate limit check passed");
                    inner.call(req).await
                }
                Err(reason) => {
                    tracing::debug!(user_id = %user_id, reason = %reason, "rate limit exceeded");
                    Ok(too_many_requests_response(&reason.to_string()))
                }
            }
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn too_many_requests_response(reason: &str) -> Response<Body> {
    let body = serde_json::json!({
        "error": "rate_limit_exceeded",
        "message": reason,
    })
    .to_string();

    Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header(http::header::CONTENT_TYPE, "application/json")
        .header("Retry-After", "60")
        .body(Body::from(body))
        .expect("static response is always valid")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;

    use axum::body::Body;
    use http::{Request, Response, StatusCode};
    use tower::{Layer, Service, ServiceExt};

    use super::*;

    // A Clone-friendly inner service for tests.
    #[derive(Clone)]
    struct OkSvc;

    impl Service<Request<Body>> for OkSvc {
        type Response = Response<Body>;
        type Error = Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(
            &mut self,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: Request<Body>) -> Self::Future {
            std::future::ready(Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::empty())
                .unwrap()))
        }
    }

    fn make_layer(rpm: u32, tpm: u32) -> RateLimitLayer {
        let (layer, _handle) = RateLimitLayer::new(rpm, tpm, std::iter::empty());
        layer
    }

    fn req_with_user(user_id: &str) -> Request<Body> {
        let mut req = Request::new(Body::empty());
        req.extensions_mut().insert(ValidatedClient {
            user_id: Some(user_id.to_owned()),
            claims: Default::default(),
        });
        req
    }

    #[tokio::test]
    async fn test_allows_request_under_limit() {
        let layer = make_layer(10, 100_000);
        let mut svc = layer.layer(OkSvc);
        svc.ready().await.unwrap();
        let resp = svc.call(req_with_user("alice")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_blocks_when_rpm_exceeded() {
        let layer = make_layer(2, 100_000);
        let mut svc = layer.layer(OkSvc);

        // Allow 2 requests.
        for _ in 0..2 {
            svc.ready().await.unwrap();
            let resp = svc.call(req_with_user("bob")).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        // 3rd request should be rate limited.
        svc.ready().await.unwrap();
        let resp = svc.call(req_with_user("bob")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_different_users_have_independent_buckets() {
        let layer = make_layer(1, 100_000);
        let mut svc = layer.layer(OkSvc);

        // alice uses her 1 request.
        svc.ready().await.unwrap();
        let resp = svc.call(req_with_user("alice")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // bob's first request should still be allowed.
        svc.ready().await.unwrap();
        let resp = svc.call(req_with_user("bob")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // alice's second request should be rate limited.
        svc.ready().await.unwrap();
        let resp = svc.call(req_with_user("alice")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_anonymous_requests_rate_limited() {
        let layer = make_layer(1, 100_000);
        let mut svc = layer.layer(OkSvc);

        // First anonymous request: ok.
        svc.ready().await.unwrap();
        let resp = svc.call(Request::new(Body::empty())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Second anonymous request: rate limited.
        svc.ready().await.unwrap();
        let resp = svc.call(Request::new(Body::empty())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_per_user_override_higher_limit() {
        let overrides = vec![(
            "power-user".to_owned(),
            RateLimitSettings {
                rpm: 100,
                tpm: 1_000_000,
            },
        )];
        let (layer, _handle) = RateLimitLayer::new(2, 100_000, overrides);
        let mut svc = layer.layer(OkSvc);

        // power-user can make 5 requests without hitting the default limit of 2.
        for i in 0..5 {
            svc.ready().await.unwrap();
            let resp = svc.call(req_with_user("power-user")).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "request {i} should be allowed"
            );
        }
    }

    #[tokio::test]
    async fn test_429_response_has_retry_after_header() {
        let layer = make_layer(0, 100_000);
        let mut svc = layer.layer(OkSvc);

        svc.ready().await.unwrap();
        let resp = svc.call(req_with_user("carol")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(resp.headers().contains_key("Retry-After"));
    }

    #[test]
    fn test_settings_for_unknown_user_returns_defaults() {
        let state = RateLimitState {
            default_rpm: 60,
            default_tpm: 100_000,
            overrides: Arc::new(DashMap::new()),
            buckets: DashMap::new(),
        };
        let settings = state.settings_for("unknown-user");
        assert_eq!(settings.rpm, 60);
        assert_eq!(settings.tpm, 100_000);
    }

    #[test]
    fn test_settings_for_known_user_returns_override() {
        let state = RateLimitState {
            default_rpm: 60,
            default_tpm: 100_000,
            overrides: Arc::new(DashMap::new()),
            buckets: DashMap::new(),
        };
        state.overrides.insert(
            "vip".to_owned(),
            RateLimitSettings {
                rpm: 600,
                tpm: 10_000_000,
            },
        );
        let settings = state.settings_for("vip");
        assert_eq!(settings.rpm, 600);
        assert_eq!(settings.tpm, 10_000_000);
    }

    #[test]
    fn test_record_tokens_updates_bucket() {
        let state = Arc::new(RateLimitState {
            default_rpm: 100,
            default_tpm: 1000,
            overrides: Arc::new(DashMap::new()),
            buckets: DashMap::new(),
        });
        // Prime the bucket.
        state.check_and_record("user1", 0).unwrap();
        // Record extra tokens.
        state.record_tokens("user1", 500);
        let bucket = state.buckets.get("user1").unwrap();
        assert_eq!(bucket.tokens, 500);
    }

    #[test]
    fn test_rate_limit_reason_display() {
        let r = RateLimitReason::Rpm {
            limit: 60,
            current: 60,
        };
        assert!(r.to_string().contains("60"));

        let r2 = RateLimitReason::Tpm {
            limit: 1000,
            current: 900,
            requested: 200,
        };
        assert!(r2.to_string().contains("1000"));

        let r3 = RateLimitReason::Anonymous;
        assert!(r3.to_string().contains("anonymous"));
    }

    // ── RateLimitHandle tests ─────────────────────────────────────────────

    #[test]
    fn test_rate_limit_handle_set_override() {
        let (_layer, handle) = RateLimitLayer::new(10, 100_000, std::iter::empty());
        handle.set_override(
            "alice",
            RateLimitSettings {
                rpm: 999,
                tpm: 5_000_000,
            },
        );
        let snap = handle.snapshot();
        assert_eq!(snap.len(), 1);
        let (id, settings) = &snap[0];
        assert_eq!(id, "alice");
        assert_eq!(settings.rpm, 999);
        assert_eq!(settings.tpm, 5_000_000);
    }

    #[test]
    fn test_rate_limit_handle_remove_override() {
        let (_layer, handle) = RateLimitLayer::new(10, 100_000, std::iter::empty());
        handle.set_override(
            "bob",
            RateLimitSettings {
                rpm: 50,
                tpm: 50_000,
            },
        );
        assert_eq!(handle.snapshot().len(), 1);
        handle.remove_override("bob");
        assert!(handle.snapshot().is_empty());
    }

    #[tokio::test]
    async fn test_rate_limit_handle_live_update_affects_service() {
        // Start with a very low rpm of 1 so the second request would normally
        // be rate-limited. Then raise the limit via the handle and verify the
        // service allows more requests.
        let (layer, handle) = RateLimitLayer::new(1, 100_000, std::iter::empty());
        let mut svc = layer.layer(OkSvc);

        // First request allowed (uses up the 1 rpm default slot for "carol").
        svc.ready().await.unwrap();
        let resp = svc.call(req_with_user("carol")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request would be blocked — but raise the limit first.
        handle.set_override(
            "carol",
            RateLimitSettings {
                rpm: 1_000,
                tpm: 10_000_000,
            },
        );

        // Now carol has a 1000 rpm override, so the second request passes.
        svc.ready().await.unwrap();
        let resp = svc.call(req_with_user("carol")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn test_rate_limit_handle_record_tokens_updates_shared_bucket() {
        // Verify that record_tokens via the handle updates the same state
        // that check_and_record uses, enabling TPM enforcement.
        let (layer, handle) = RateLimitLayer::new(100, 50, std::iter::empty());
        let state = Arc::clone(&layer.inner);

        // Prime the bucket by checking and recording a request (0 tokens).
        state.check_and_record("dave", 0).unwrap();
        // Record 30 tokens via handle (simulating post-response token accounting).
        handle.record_tokens("dave", 30);
        // After 30 tokens, another request can still pass (30 <= 50).
        state.check_and_record("dave", 0).unwrap();
        // Record 25 more tokens (total now 55 > 50).
        handle.record_tokens("dave", 25);
        // Now check should fail: 55 + 0 > 50.
        let result = state.check_and_record("dave", 0);
        assert!(
            matches!(result, Err(RateLimitReason::Tpm { .. })),
            "expected TPM rate limit after accumulating 55 tokens with tpm=50"
        );
    }
}
