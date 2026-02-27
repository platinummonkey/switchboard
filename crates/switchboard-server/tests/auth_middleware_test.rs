//! Integration tests for the 401/200 auth flow through the full Tower middleware stack.
//!
//! Unlike `auth_test.rs` and `proxy_test.rs` — which inject a `ValidatedClient`
//! extension directly to bypass the auth layer — these tests wire up the FULL
//! `build_middleware_stack` stack so that every request passes through:
//!
//! 1. `RequestIdLayer`
//! 2. `ModelOverrideLayer`
//! 3. `AuthLayer`      ← the focus of these tests
//! 4. `RateLimitLayer`
//! 5. `AuthInjectLayer`
//!
//! This lets us verify the end-to-end 401/200 flow without mocking any
//! individual layer.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use http::{HeaderName, HeaderValue};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use switchboard_server::auth::{AuthRegistryBuilder, StaticKeyValidator, UpstreamCredentials};
use switchboard_server::config::ServerConfig;
use switchboard_server::config::model_selection::ModelSelectionConfig;
use switchboard_server::config::provider::{KeyPoolConfig, ProviderConfig};
use switchboard_server::key_pool::health::KeyHealth;
use switchboard_server::key_pool::pool::KeyPool;
use switchboard_server::key_pool::provider::{KeySource, PooledKey};
use switchboard_server::key_pool::selector::WeightedRandomSelector;
use switchboard_server::middleware::{
    MiddlewareConfig, RateLimitLayer, RateLimitSettings, build_middleware_stack,
};
use switchboard_server::providers::{OpenAiProvider, ProviderRegistry};
use switchboard_server::proxy::handler::{AppState, chat_completions, health};
use switchboard_server::routing::ModelSelector;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_upstream_key(header_name: &'static str, header_value: &'static str) -> PooledKey {
    PooledKey {
        id: "middleware-test-key".into(),
        credentials: UpstreamCredentials {
            header_name: HeaderName::from_static(header_name),
            header_value: HeaderValue::from_static(header_value),
            expires_at: None,
        },
        weight: 1.0,
        source: KeySource::Static,
        health: KeyHealth::default(),
    }
}

fn make_pool(key: PooledKey) -> Arc<KeyPool> {
    Arc::new(KeyPool::new(vec![key], Box::new(WeightedRandomSelector)))
}

/// Build an `AppState` pointing OpenAI at the given mock server URL.
fn openai_app_state(mock_url: &str) -> Arc<AppState> {
    let provider = Arc::new(OpenAiProvider::new_named(
        "openai",
        mock_url,
        vec!["gpt-4o".into(), "gpt-4o-mini".into()],
        Duration::from_secs(10),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("openai", provider);

    let mut key_pools = HashMap::new();
    key_pools.insert(
        "openai".into(),
        make_pool(make_upstream_key("authorization", "Bearer sk-upstream")),
    );

    let mut config = ServerConfig::default();
    config.providers.insert(
        "openai".into(),
        ProviderConfig {
            base_url: Some(mock_url.into()),
            api_format: "openai".into(),
            models: vec!["gpt-4o".into(), "gpt-4o-mini".into()],
            region: None,
            cross_region_inference: false,
            project_id: None,
            timeout: "10s".into(),
            health_check_interval: "30s".into(),
            max_concurrent: 10,
            key_pool: KeyPoolConfig::default(),
        },
    );

    Arc::new(AppState {
        config: Arc::new(config),
        providers: Arc::new(registry),
        key_pools: Arc::new(key_pools),
        usage: Arc::new(switchboard_server::observability::UsageTracker::new()),
    })
}

/// Build an `AppState` with no providers (for pure auth tests where we only
/// care about the 401 response and never reach the proxy handler).
fn minimal_app_state() -> Arc<AppState> {
    Arc::new(AppState {
        config: Arc::new(ServerConfig::default()),
        providers: Arc::new(ProviderRegistry::new()),
        key_pools: Arc::new(HashMap::new()),
        usage: Arc::new(switchboard_server::observability::UsageTracker::new()),
    })
}

/// Build a router with the FULL middleware stack using the provided client keys.
///
/// The `/health` route is placed on a separate (unauthenticated) sub-router so
/// it bypasses the auth layer — mirroring the intended production behaviour
/// where health checks must work without credentials.  All other routes are
/// wrapped by the full stack.
///
/// `valid_keys` — the static API keys that the `StaticKeyValidator` will accept.
/// `default_rpm` — RPM limit used for the rate-limit layer.
fn build_authed_router(
    valid_keys: Vec<String>,
    app_state: Arc<AppState>,
    default_rpm: u32,
) -> Router {
    let validator = StaticKeyValidator::new("test", valid_keys);
    let registry = AuthRegistryBuilder::default().add(validator).build();

    let key_pools: Arc<HashMap<String, Arc<KeyPool>>> = Arc::clone(&app_state.key_pools);

    let providers_config = Arc::new(app_state.config.providers.clone());
    let provider_registry = Arc::clone(&app_state.providers);

    let (stack, _handle) = build_middleware_stack(MiddlewareConfig {
        auth_registry: Arc::new(registry),
        default_rpm,
        default_tpm: 10_000_000,
        rate_limit_overrides: vec![],
        key_pools,
        model_selector: Arc::new(ModelSelector::new(ModelSelectionConfig::default())),
        provider_registry,
        providers_config,
        guardrail_pipeline: None,
        rate_limit_layer: None,
    });

    // Authenticated routes — full middleware stack applied.
    let authed = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(Arc::clone(&app_state))
        .layer(stack);

    // Unauthenticated routes — no auth layer.
    // Note: in this test setup `/health` is intentionally placed OUTSIDE the
    // authenticated sub-router so it is accessible without credentials.
    Router::new().route("/health", get(health)).merge(authed)
}

/// Build a router where ALL routes (including `/health`) pass through the
/// auth middleware — used by the test that verifies `/health` returns 401
/// when placed behind auth.
fn build_fully_authed_router(valid_keys: Vec<String>, app_state: Arc<AppState>) -> Router {
    let validator = StaticKeyValidator::new("test-all", valid_keys);
    let registry = AuthRegistryBuilder::default().add(validator).build();

    let key_pools: Arc<HashMap<String, Arc<KeyPool>>> = Arc::clone(&app_state.key_pools);

    let providers_config = Arc::new(app_state.config.providers.clone());
    let provider_registry = Arc::clone(&app_state.providers);

    let (stack, _handle) = build_middleware_stack(MiddlewareConfig {
        auth_registry: Arc::new(registry),
        default_rpm: 1000,
        default_tpm: 10_000_000,
        rate_limit_overrides: vec![],
        key_pools,
        model_selector: Arc::new(ModelSelector::new(ModelSelectionConfig::default())),
        provider_registry,
        providers_config,
        guardrail_pipeline: None,
        rate_limit_layer: None,
    });

    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(app_state)
        .layer(stack)
}

/// Build a standard JSON chat-completion POST request body.
fn chat_body() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    }))
    .unwrap()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Valid `Authorization: Bearer <key>` header + wiremock upstream → 200.
///
/// Verifies the happy path: a correct bearer token passes through the auth
/// layer and the request is successfully proxied to the upstream.
#[tokio::test]
async fn test_auth_middleware_valid_key_returns_200() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-auth-ok",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "authenticated!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = openai_app_state(&mock_server.uri());
    let app = build_authed_router(vec!["sk-valid-key".into()], state, 1000);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-valid-key")
        .body(Body::from(chat_body()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), 8192).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "authenticated!");
}

/// Missing `Authorization` header → 401 Unauthorized.
///
/// The `AuthLayer` must reject requests that carry no credentials at all.
#[tokio::test]
async fn test_auth_middleware_missing_auth_returns_401() {
    let state = minimal_app_state();
    let app = build_authed_router(vec!["sk-any".into()], state, 1000);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_body()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// Wrong bearer token → 401 Unauthorized.
///
/// Verifies that a request with a syntactically valid but incorrect API key
/// is rejected before reaching the handler.
#[tokio::test]
async fn test_auth_middleware_wrong_key_returns_401() {
    let state = minimal_app_state();
    let app = build_authed_router(vec!["sk-correct-key".into()], state, 1000);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-wrong-key")
        .body(Body::from(chat_body()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// When ALL routes share the auth layer, `GET /health` also returns 401
/// without credentials.
///
/// Note: the `build_authed_router` helper intentionally places `/health`
/// OUTSIDE the auth-protected sub-router to match production behaviour.  This
/// test uses `build_fully_authed_router` — a variant where every route is
/// wrapped by the auth layer — to confirm the middleware behaves correctly in
/// that configuration.
#[tokio::test]
async fn test_auth_middleware_health_behind_auth_returns_401() {
    let state = minimal_app_state();
    // Fully-authed router: `/health` is behind auth.
    let app = build_fully_authed_router(vec!["sk-any".into()], state);

    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// `GET /health` on the production-style router (health outside auth) returns
/// 200 even without credentials.
///
/// Confirms the correct production routing structure: unauthenticated health
/// checks must succeed.
#[tokio::test]
async fn test_auth_middleware_health_bypasses_auth_returns_200() {
    let state = minimal_app_state();
    let app = build_authed_router(vec!["sk-any".into()], state, 1000);

    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["status"], "ok");
}

/// `Authorization: Bearer sk-test` where the stored key is `sk-test`
/// (not `Bearer sk-test`) → 200.
///
/// Confirms `StaticKeyValidator` correctly strips the `Bearer ` prefix before
/// comparing against the configured key list.
#[tokio::test]
async fn test_auth_middleware_bearer_prefix_stripped() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-bearer-strip",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "prefix stripped"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = openai_app_state(&mock_server.uri());
    // Key stored without prefix — validator must strip "Bearer " from the header.
    let app = build_authed_router(vec!["sk-test".into()], state, 1000);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        // Header includes "Bearer " prefix; stored key is just "sk-test".
        .header("authorization", "Bearer sk-test")
        .body(Body::from(chat_body()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "prefix stripped");
}

/// RPM=1, two requests in quick succession → second request returns 429.
///
/// Builds the middleware with a very tight rate limit and verifies that the
/// `RateLimitLayer` fires correctly after the first request is admitted.
///
/// Note: both requests carry a valid API key so the 429 is definitely from
/// the rate limiter, not the auth layer.
#[tokio::test]
async fn test_auth_middleware_rate_limit_429() {
    let state = minimal_app_state();

    let validator = StaticKeyValidator::new("test-rl", vec!["sk-rl-key"]);
    let registry = AuthRegistryBuilder::default().add(validator).build();

    let key_pools: Arc<HashMap<String, Arc<KeyPool>>> = Arc::clone(&state.key_pools);

    // Pre-build the rate-limit layer with RPM=1 so we can share it across
    // the two oneshot calls (same bucket state).
    let (rl_layer, _handle) = RateLimitLayer::new(
        1,
        10_000_000,
        std::iter::empty::<(String, RateLimitSettings)>(),
    );

    let providers_config = Arc::new(state.config.providers.clone());
    let provider_registry = Arc::clone(&state.providers);

    let (stack, _stack_handle) = build_middleware_stack(MiddlewareConfig {
        auth_registry: Arc::new(registry),
        default_rpm: 1,
        default_tpm: 10_000_000,
        rate_limit_overrides: vec![],
        key_pools,
        model_selector: Arc::new(ModelSelector::new(ModelSelectionConfig::default())),
        provider_registry,
        providers_config,
        guardrail_pipeline: None,
        rate_limit_layer: Some((rl_layer, _handle)),
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(stack);

    let make_req = || {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer sk-rl-key")
            .body(Body::from(chat_body()))
            .unwrap()
    };

    // First request: should pass auth but may return a non-200 from the proxy
    // (no provider is configured) — what matters is it's NOT a 429.
    let resp1 = app.clone().oneshot(make_req()).await.unwrap();
    assert_ne!(
        resp1.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "first request should not be rate-limited"
    );

    // Second request within the same minute window: rate limiter must fire.
    let resp2 = app.oneshot(make_req()).await.unwrap();
    assert_eq!(
        resp2.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "second request should be rate-limited (RPM=1)"
    );
}

/// Unauthorized request → response body is valid JSON with an `"error"` field.
///
/// Callers relying on JSON error bodies must receive a parseable response even
/// when auth fails.
#[tokio::test]
async fn test_auth_middleware_401_response_is_json() {
    let state = minimal_app_state();
    let app = build_authed_router(vec!["sk-any".into()], state, 1000);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        // No Authorization header.
        .body(Body::from(chat_body()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // Content-Type must be application/json.
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("application/json"),
        "401 response must have application/json content-type, got: {ct}"
    );

    // Body must be valid JSON containing an "error" key.
    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value =
        serde_json::from_slice(&bytes).expect("401 body must be valid JSON");
    assert!(
        json.get("error").is_some(),
        "401 JSON body must contain an \"error\" field, got: {json}"
    );
}

/// Successful auth injects a `ValidatedClient` that the handler can access.
///
/// Uses a custom axum handler that reads the `ValidatedClient` extension and
/// echoes its presence back in the response, ensuring the auth layer wired the
/// extension correctly end-to-end.
#[tokio::test]
async fn test_auth_middleware_valid_key_injects_validated_client() {
    use axum::Extension;
    use switchboard_server::auth::ValidatedClient;

    /// Handler that reads the `ValidatedClient` extension and returns 200 with
    /// `{"authenticated": true}` when present, or 500 when absent.
    async fn echo_auth(
        Extension(client): Extension<ValidatedClient>,
    ) -> impl axum::response::IntoResponse {
        let _ = client; // just check it's present
        axum::Json(serde_json::json!({"authenticated": true}))
    }

    let validator = StaticKeyValidator::new("test-inject", vec!["sk-inject-key"]);
    let registry = AuthRegistryBuilder::default().add(validator).build();

    let state = minimal_app_state();
    let key_pools: Arc<HashMap<String, Arc<KeyPool>>> = Arc::clone(&state.key_pools);

    let providers_config = Arc::new(state.config.providers.clone());
    let provider_registry = Arc::clone(&state.providers);

    let (stack, _handle) = build_middleware_stack(MiddlewareConfig {
        auth_registry: Arc::new(registry),
        default_rpm: 1000,
        default_tpm: 10_000_000,
        rate_limit_overrides: vec![],
        key_pools,
        model_selector: Arc::new(ModelSelector::new(ModelSelectionConfig::default())),
        provider_registry,
        providers_config,
        guardrail_pipeline: None,
        rate_limit_layer: None,
    });

    let app = Router::new().route("/probe", get(echo_auth)).layer(stack);

    let req = Request::builder()
        .method("GET")
        .uri("/probe")
        .header("authorization", "Bearer sk-inject-key")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "handler should receive ValidatedClient extension and return 200"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        json["authenticated"], true,
        "handler should confirm ValidatedClient was injected"
    );
}
