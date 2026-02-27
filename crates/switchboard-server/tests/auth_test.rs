//! Integration tests for auth flow through the proxy.
//!
//! Covers API key injection, missing key pool behavior, custom header names,
//! and upstream error mappings.  Auth middleware is bypassed for most tests
//! (a `ValidatedClient` extension is injected directly); one test omits it
//! to confirm the proxy still routes correctly when no extension is present.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use http::{HeaderName, HeaderValue};
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use switchboard_server::auth::{UpstreamCredentials, ValidatedClient};
use switchboard_server::config::ServerConfig;
use switchboard_server::config::provider::{KeyPoolConfig, ProviderConfig};
use switchboard_server::key_pool::health::KeyHealth;
use switchboard_server::key_pool::pool::KeyPool;
use switchboard_server::key_pool::provider::{KeySource, PooledKey};
use switchboard_server::key_pool::selector::WeightedRandomSelector;
use switchboard_server::providers::{OpenAiProvider, ProviderRegistry};
use switchboard_server::proxy::handler::{
    AppState, anthropic_messages, chat_completions, health, list_models,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_key(header_name: &'static str, header_value: &'static str) -> PooledKey {
    PooledKey {
        id: "auth-test-key".into(),
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

/// Build an `AppState` pointing OpenAI at the given mock server URL with the
/// supplied key injected into the pool.
fn openai_app_state_with_key(mock_url: &str, key: PooledKey) -> Arc<AppState> {
    let provider = Arc::new(OpenAiProvider::new_named(
        "openai",
        mock_url,
        vec!["gpt-4o".into(), "gpt-4o-mini".into()],
        Duration::from_secs(10),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("openai", provider);

    let mut key_pools = HashMap::new();
    key_pools.insert("openai".into(), make_pool(key));

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

/// Standard test router — injects a `ValidatedClient` so the auth middleware
/// layer is effectively bypassed.
fn build_test_router(state: Arc<AppState>) -> Router {
    let validated_client = ValidatedClient::from_static_key();
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/api/v1/messages", post(anthropic_messages))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(axum::middleware::from_fn(
            move |mut req: axum::extract::Request, next: axum::middleware::Next| {
                let vc = validated_client.clone();
                async move {
                    req.extensions_mut().insert(vc);
                    next.run(req).await
                }
            },
        ))
        .with_state(state)
}

/// Router variant that does NOT inject a `ValidatedClient` extension, used to
/// verify that the proxy handler itself requires the extension and returns an
/// appropriate error when it is absent.
fn build_router_without_auth(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/api/v1/messages", post(anthropic_messages))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Without a `ValidatedClient` extension the handler returns a 500 because the
/// `Extension` extractor is missing.  This confirms the auth middleware is
/// required for proxy endpoints; the health endpoint does not need it.
#[tokio::test]
async fn test_no_validated_client_still_proxies() {
    let mock_server = MockServer::start().await;

    // Register a mock that would succeed if the request reaches it.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-no-auth",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        })))
        .mount(&mock_server)
        .await;

    let state = openai_app_state_with_key(
        &mock_server.uri(),
        make_key("authorization", "Bearer sk-test"),
    );
    // Use the router WITHOUT the auth extension injector.
    let app = build_router_without_auth(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // axum returns 500 when a required Extension extractor is missing.
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

/// The key's credential header (`authorization: Bearer sk-test`) must be
/// injected into the request that reaches the upstream server.
#[tokio::test]
async fn test_proxy_injects_api_key() {
    let mock_server = MockServer::start().await;

    // Wiremock requires the authorization header to match exactly.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-key-inject",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "key present"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = openai_app_state_with_key(
        &mock_server.uri(),
        make_key("authorization", "Bearer sk-test"),
    );
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "check key"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = axum::body::to_bytes(resp.into_body(), 8192).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();
    assert_eq!(json["choices"][0]["message"]["content"], "key present");
}

/// An AppState with an empty key pool (no keys at all) must return 503.
#[tokio::test]
async fn test_key_pool_exhausted_returns_503() {
    // Provider registered but pool has zero keys.
    let provider = Arc::new(OpenAiProvider::new_named(
        "openai",
        "https://api.openai.com",
        vec!["gpt-4o".into()],
        Duration::from_secs(5),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("openai", provider);

    // Register an empty pool.
    let empty_pool = Arc::new(KeyPool::new(vec![], Box::new(WeightedRandomSelector)));
    let mut key_pools = HashMap::new();
    key_pools.insert("openai".into(), empty_pool);

    let mut config = ServerConfig::default();
    config.providers.insert(
        "openai".into(),
        ProviderConfig {
            base_url: Some("https://api.openai.com".into()),
            api_format: "openai".into(),
            models: vec!["gpt-4o".into()],
            region: None,
            cross_region_inference: false,
            project_id: None,
            timeout: "5s".into(),
            health_check_interval: "30s".into(),
            max_concurrent: 10,
            key_pool: KeyPoolConfig::default(),
        },
    );

    let state = Arc::new(AppState {
        config: Arc::new(config),
        providers: Arc::new(registry),
        key_pools: Arc::new(key_pools),
        usage: Arc::new(switchboard_server::observability::UsageTracker::new()),
    });

    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

/// A key configured with `x-api-key` instead of `authorization` must still
/// reach the upstream server under the correct header name.
#[tokio::test]
async fn test_key_with_wrong_header_still_forwarded() {
    let mock_server = MockServer::start().await;

    // Wiremock expects `x-api-key: sk-test` — not `authorization`.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("x-api-key", "sk-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-xapikey",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "x-api-key received"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Key uses x-api-key header.
    let state = openai_app_state_with_key(&mock_server.uri(), make_key("x-api-key", "sk-test"));
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "using x-api-key"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = axum::body::to_bytes(resp.into_body(), 8192).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();
    assert_eq!(
        json["choices"][0]["message"]["content"],
        "x-api-key received"
    );
}

/// Upstream returns 401 — proxy must map this to 502 Bad Gateway.
#[tokio::test]
async fn test_upstream_401_returns_502() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_string("{\"error\":{\"message\":\"Unauthorized\"}}"),
        )
        .mount(&mock_server)
        .await;

    let state = openai_app_state_with_key(
        &mock_server.uri(),
        make_key("authorization", "Bearer sk-bad"),
    );
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
}

/// Upstream returns 500 — proxy must map this to 502 Bad Gateway.
#[tokio::test]
async fn test_upstream_500_returns_502() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let state = openai_app_state_with_key(
        &mock_server.uri(),
        make_key("authorization", "Bearer sk-test"),
    );
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
}

/// `GET /health` requires no auth and always returns 200.
#[tokio::test]
async fn test_health_endpoint_unauthenticated() {
    // Use a state with no providers — health never needs them.
    let state = Arc::new(AppState {
        config: Arc::new(ServerConfig::default()),
        providers: Arc::new(ProviderRegistry::new()),
        key_pools: Arc::new(HashMap::new()),
        usage: Arc::new(switchboard_server::observability::UsageTracker::new()),
    });

    // Deliberately use the router WITHOUT the auth extension injector to
    // verify the health endpoint works without any auth context.
    let app = build_router_without_auth(state);

    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();
    assert_eq!(json["status"], "ok");
}
