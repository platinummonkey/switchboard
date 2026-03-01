//! Integration tests for SSE streaming through the proxy.
//!
//! Each test spins up a wiremock server that returns `text/event-stream`
//! responses, sends a request through the full handler stack, and verifies
//! that the streaming body is forwarded correctly to the client.

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

use switchboard_server::auth::{UpstreamCredentials, ValidatedClient};
use switchboard_server::config::ServerConfig;
use switchboard_server::config::provider::{KeyPoolConfig, ProviderConfig};
use switchboard_server::key_pool::health::KeyHealth;
use switchboard_server::key_pool::pool::KeyPool;
use switchboard_server::key_pool::provider::{KeySource, PooledKey};
use switchboard_server::key_pool::selector::WeightedRandomSelector;
use switchboard_server::providers::{AnthropicProvider, OpenAiProvider, ProviderRegistry};
use switchboard_server::proxy::handler::{
    AppState, anthropic_messages, chat_completions, health, list_models,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_key(header_name: &'static str, header_value: &'static str) -> PooledKey {
    PooledKey {
        id: "streaming-key".into(),
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
        make_pool(make_key("authorization", "Bearer sk-test")),
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
        rate_limit_handle: None,
    })
}

fn anthropic_app_state(mock_url: &str) -> Arc<AppState> {
    let provider = Arc::new(AnthropicProvider::new_with_base_url(
        mock_url,
        vec![
            "claude-sonnet-4-20250514".into(),
            "claude-opus-4-20250514".into(),
        ],
        Duration::from_secs(10),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("anthropic", provider);

    let mut key_pools = HashMap::new();
    key_pools.insert(
        "anthropic".into(),
        make_pool(make_key("x-api-key", "sk-ant-test")),
    );

    let mut config = ServerConfig::default();
    config.providers.insert(
        "anthropic".into(),
        ProviderConfig {
            base_url: Some(mock_url.into()),
            api_format: "anthropic".into(),
            models: vec![
                "claude-sonnet-4-20250514".into(),
                "claude-opus-4-20250514".into(),
            ],
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
        rate_limit_handle: None,
    })
}

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

// ── Tests ─────────────────────────────────────────────────────────────────────

/// OpenAI streaming round-trip: wiremock returns `text/event-stream`, proxy
/// forwards it to the client with the same content-type.
#[tokio::test]
async fn test_openai_streaming_response() {
    let mock_server = MockServer::start().await;

    let sse_body = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = openai_app_state(&mock_server.uri());
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    // The proxy must set content-type to text/event-stream.
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "expected SSE content-type, got: {content_type}"
    );

    // Read body and verify SSE events are present.
    let resp_bytes = axum::body::to_bytes(resp.into_body(), 65536).await.unwrap();
    let body_str = std::str::from_utf8(&resp_bytes).unwrap();
    assert!(
        body_str.contains("data:"),
        "expected SSE data lines in body, got: {body_str}"
    );
}

/// Anthropic streaming round-trip: wiremock returns `text/event-stream`, proxy
/// forwards it with SSE content-type.
#[tokio::test]
async fn test_anthropic_streaming_response() {
    let mock_server = MockServer::start().await;

    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = anthropic_app_state(&mock_server.uri());
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 256,
        "stream": true
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "expected SSE content-type, got: {content_type}"
    );

    let resp_bytes = axum::body::to_bytes(resp.into_body(), 65536).await.unwrap();
    let body_str = std::str::from_utf8(&resp_bytes).unwrap();
    assert!(
        body_str.contains("data:"),
        "expected SSE data lines in body, got: {body_str}"
    );
}

/// Five-chunk streaming: all 5 SSE data events must reach the client.
#[tokio::test]
async fn test_streaming_with_multiple_chunks() {
    let mock_server = MockServer::start().await;

    // Build a response body with 5 distinct SSE chunks.
    let mut sse_body = String::new();
    for i in 0..5 {
        sse_body.push_str(&format!(
            "data: {{\"id\":\"{i}\",\"choices\":[{{\"delta\":{{\"content\":\"chunk{i}\"}},\"finish_reason\":null}}]}}\n\n"
        ));
    }
    sse_body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = openai_app_state(&mock_server.uri());
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "count to 5"}],
        "stream": true
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = axum::body::to_bytes(resp.into_body(), 65536).await.unwrap();
    let body_str = std::str::from_utf8(&resp_bytes).unwrap();

    // Verify all 5 chunks arrived.
    for i in 0..5 {
        assert!(
            body_str.contains(&format!("chunk{i}")),
            "expected chunk{i} in streaming body, got: {body_str}"
        );
    }
}

/// Non-streaming request with `stream: false` returns regular JSON, not SSE.
#[tokio::test]
async fn test_non_streaming_still_works() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-non-stream",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Non-streaming response."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = openai_app_state(&mock_server.uri());
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Response content-type must NOT be SSE.
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        !content_type.contains("text/event-stream"),
        "non-streaming response should not be SSE, got content-type: {content_type}"
    );

    // Body must be valid JSON (not raw SSE).
    let resp_bytes = axum::body::to_bytes(resp.into_body(), 65536).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&resp_bytes).expect("expected JSON body");
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(
        json["choices"][0]["message"]["content"],
        "Non-streaming response."
    );
}
