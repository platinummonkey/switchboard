//! Integration tests for the proxy engine.
//!
//! Each test spins up a wiremock server to simulate upstream providers,
//! builds an `AppState`, constructs the axum router, and makes real HTTP
//! requests through the full handler stack.

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
use switchboard_server::providers::{
    AnthropicProvider, BedrockProvider, OllamaProvider, OpenAiProvider, ProviderRegistry,
    VertexProvider,
};
use switchboard_server::proxy::handler::{
    AppState, anthropic_messages, chat_completions, health, list_models,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_key(header_name: &'static str, header_value: &'static str) -> PooledKey {
    PooledKey {
        id: "integration-key".into(),
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
    })
}

/// Build an `AppState` pointing Anthropic at the given mock server URL.
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
    })
}

/// Build an axum router from an `AppState`, injecting a bypass `ValidatedClient`
/// so the auth middleware is skipped in tests.
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

/// Non-streaming OpenAI round-trip through the proxy.
#[tokio::test]
async fn test_openai_proxy_non_streaming() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-integration",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "The answer is 42."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = openai_app_state(&mock_server.uri());
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"}
        ],
        "max_tokens": 128
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

    // Verify the response is in OpenAI format.
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(
        json["choices"][0]["message"]["content"],
        "The answer is 42."
    );
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 12);
    assert_eq!(json["usage"]["completion_tokens"], 7);
    assert_eq!(json["usage"]["total_tokens"], 19);
}

/// Non-streaming Anthropic round-trip through the proxy.
#[tokio::test]
async fn test_anthropic_proxy_non_streaming() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_integration",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = anthropic_app_state(&mock_server.uri());
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "user", "content": "Hello, Claude!"}
        ],
        "max_tokens": 256
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = axum::body::to_bytes(resp.into_body(), 8192).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();

    // Verify the response is in Anthropic format.
    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "assistant");
    assert_eq!(json["content"][0]["type"], "text");
    assert_eq!(json["content"][0]["text"], "Hello from Claude!");
    assert_eq!(json["stop_reason"], "end_turn");
    assert_eq!(json["usage"]["input_tokens"], 10);
    assert_eq!(json["usage"]["output_tokens"], 5);
}

/// Model list endpoint returns registered models.
#[tokio::test]
async fn test_list_models() {
    let state = openai_app_state("https://api.openai.com");
    let app = build_test_router(state);

    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = axum::body::to_bytes(resp.into_body(), 8192).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();

    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().unwrap();
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"gpt-4o"), "gpt-4o should be in models list");
    assert!(
        ids.contains(&"gpt-4o-mini"),
        "gpt-4o-mini should be in models list"
    );
}

/// Health endpoint returns 200 with status ok.
#[tokio::test]
async fn test_health_endpoint() {
    let state = Arc::new(AppState {
        config: Arc::new(ServerConfig::default()),
        providers: Arc::new(ProviderRegistry::new()),
        key_pools: Arc::new(HashMap::new()),
    });
    let app = build_test_router(state);

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

/// Unknown model returns 400 Bad Request.
#[tokio::test]
async fn test_unknown_model_returns_400() {
    let state = openai_app_state("https://api.openai.com");
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "nonexistent-model-xyz",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let resp_bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("nonexistent-model-xyz")
    );
}

/// Upstream 5xx error is returned as 502 Bad Gateway.
#[tokio::test]
async fn test_upstream_5xx_returns_502() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("Service Unavailable"))
        .mount(&mock_server)
        .await;

    let state = openai_app_state(&mock_server.uri());
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

/// Request with system message is forwarded correctly.
#[tokio::test]
async fn test_openai_request_with_system_message() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-sys",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Arr, I be a pirate!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
        })))
        .mount(&mock_server)
        .await;

    let state = openai_app_state(&mock_server.uri());
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
            {"role": "user", "content": "Hello!"}
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp_bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();
    assert_eq!(
        json["choices"][0]["message"]["content"],
        "Arr, I be a pirate!"
    );
}

/// Anthropic request with no key pool returns 503.
#[tokio::test]
async fn test_no_key_pool_returns_503() {
    // Build state with a provider registered but no key pool.
    let provider = Arc::new(AnthropicProvider::new_with_base_url(
        "https://api.anthropic.com",
        vec!["claude-sonnet-4-20250514".into()],
        Duration::from_secs(5),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("anthropic", provider);

    // No key pools registered.
    let key_pools: HashMap<String, Arc<KeyPool>> = HashMap::new();

    let mut config = ServerConfig::default();
    config.providers.insert(
        "anthropic".into(),
        ProviderConfig {
            base_url: Some("https://api.anthropic.com".into()),
            api_format: "anthropic".into(),
            models: vec!["claude-sonnet-4-20250514".into()],
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
    });

    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 100
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

// ── Phase 8: Bedrock, Vertex, Ollama integration tests ────────────────────────

/// Helper: build an `AppState` with VertexProvider pointing at a mock server.
fn vertex_app_state(mock_url: &str) -> Arc<AppState> {
    let provider = Arc::new(VertexProvider::new_with_params(
        "us-central1",
        "test-project",
        vec!["gemini-1.5-pro".into()],
        Duration::from_secs(10),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("vertex", provider);

    let mut key_pools = HashMap::new();
    key_pools.insert(
        "vertex".into(),
        make_pool(make_key("authorization", "Bearer ya29.test")),
    );

    let mut config = ServerConfig::default();
    config.providers.insert(
        "vertex".into(),
        ProviderConfig {
            base_url: Some(mock_url.into()),
            api_format: "vertex".into(),
            models: vec!["gemini-1.5-pro".into()],
            region: Some("us-central1".into()),
            cross_region_inference: false,
            project_id: Some("test-project".into()),
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
    })
}

/// Helper: build an `AppState` with OllamaProvider pointing at a mock server.
fn ollama_app_state(mock_url: &str) -> Arc<AppState> {
    let provider = Arc::new(OllamaProvider::new_with_base_url(
        mock_url,
        vec!["llama3.2".into(), "mistral".into()],
        Duration::from_secs(10),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("ollama", provider);

    let mut key_pools = HashMap::new();
    key_pools.insert(
        "ollama".into(),
        make_pool(make_key("authorization", "Bearer ollama")),
    );

    let mut config = ServerConfig::default();
    config.providers.insert(
        "ollama".into(),
        ProviderConfig {
            base_url: Some(mock_url.into()),
            api_format: "ollama".into(),
            models: vec!["llama3.2".into(), "mistral".into()],
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
    })
}

/// Non-streaming Bedrock Converse API round-trip through the proxy.
///
/// Bedrock uses SigV4 signing and constructs provider-specific URLs
/// (https://bedrock-runtime.{region}.amazonaws.com/…). We cannot redirect the
/// provider to wiremock at the proxy-handler level because BedrockProvider
/// builds its own URL internally. Instead this test verifies:
///   1. BedrockProvider is correctly registered in the provider registry.
///   2. Model routing resolves `anthropic.*`, `amazon.*`, `meta.*`, `mistral.*`
///      prefixes to BedrockProvider.
///   3. Provider metadata (name, api_format) is correct.
///
/// The unit-level network tests in `providers/bedrock.rs` cover SigV4 signing
/// and actual HTTP interaction with a mock server.
#[tokio::test]
async fn test_bedrock_non_streaming() {
    let creds_json = serde_json::json!({
        "access_key": "AKIATEST",
        "secret_key": "testsecret",
        "session_token": "",
        "region": "us-east-1"
    });

    let provider = Arc::new(BedrockProvider::new_with_params(
        "us-east-1",
        false,
        vec!["anthropic.claude-3-sonnet".into()],
        Duration::from_secs(10),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("bedrock", provider);

    let key = PooledKey {
        id: "bedrock-key".into(),
        credentials: UpstreamCredentials {
            header_name: HeaderName::from_static("x-switchboard-bedrock-creds"),
            header_value: HeaderValue::from_str(&creds_json.to_string()).unwrap(),
            expires_at: None,
        },
        weight: 1.0,
        source: KeySource::Static,
        health: KeyHealth::default(),
    };
    let mut key_pools = HashMap::new();
    key_pools.insert("bedrock".into(), make_pool(key));

    let mut config = ServerConfig::default();
    config.providers.insert(
        "bedrock".into(),
        ProviderConfig {
            base_url: None,
            api_format: "bedrock".into(),
            models: vec!["anthropic.claude-3-sonnet".into()],
            region: Some("us-east-1".into()),
            cross_region_inference: false,
            project_id: None,
            timeout: "10s".into(),
            health_check_interval: "30s".into(),
            max_concurrent: 10,
            key_pool: KeyPoolConfig::default(),
        },
    );

    let state = Arc::new(AppState {
        config: Arc::new(config),
        providers: Arc::new(registry),
        key_pools: Arc::new(key_pools),
    });

    // Verify that the BedrockProvider is correctly registered and resolves the
    // model.  The actual network call to the provider will fail (it tries to
    // reach AWS) but we test routing by checking supports_model and registration.
    let providers_ref = &state.providers;
    let providers_config = &state.config.providers;
    let resolved = providers_ref.resolve_provider("anthropic.claude-3-sonnet", providers_config);
    assert!(
        resolved.is_some(),
        "BedrockProvider should resolve anthropic.claude-3-sonnet"
    );
    let (prov, _) = resolved.unwrap();
    assert_eq!(prov.name(), "bedrock");
    assert_eq!(prov.api_format(), "bedrock");
    assert!(prov.supports_model("anthropic.claude-3-sonnet"));
    assert!(prov.supports_model("amazon.titan-text-express-v1"));
    assert!(prov.supports_model("meta.llama3-8b-instruct-v1:0"));
    assert!(prov.supports_model("mistral.mistral-7b-instruct-v0:2"));
}

/// Non-streaming Vertex AI Gemini round-trip — verifies provider registration
/// and model resolution.
#[tokio::test]
async fn test_vertex_non_streaming() {
    let state = vertex_app_state("https://us-central1-aiplatform.googleapis.com");

    // Verify registration and routing.
    let resolved = state
        .providers
        .resolve_provider("gemini-1.5-pro", &state.config.providers);
    assert!(
        resolved.is_some(),
        "VertexProvider should resolve gemini-1.5-pro"
    );
    let (prov, _) = resolved.unwrap();
    assert_eq!(prov.name(), "vertex");
    assert_eq!(prov.api_format(), "vertex");
    assert!(prov.supports_model("gemini-1.5-pro"));
    assert!(prov.supports_model("gemini-2.0-flash-exp"));
    assert!(!prov.supports_model("gpt-4o"));
}

/// Ollama provider delegates to OpenAI-compatible `/v1/chat/completions` endpoint.
///
/// This test makes a real HTTP request through the full proxy pipeline to
/// confirm Ollama is wired correctly and uses the OpenAI endpoint path.
#[tokio::test]
async fn test_ollama_non_streaming() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-ollama-integration",
            "object": "chat.completion",
            "model": "llama3.2",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Ollama!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let state = ollama_app_state(&mock_server.uri());
    let app = build_test_router(state);

    let body = serde_json::json!({
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "Hello, Ollama!"}
        ],
        "max_tokens": 128
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

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(
        json["choices"][0]["message"]["content"],
        "Hello from Ollama!"
    );
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 8);
    assert_eq!(json["usage"]["completion_tokens"], 4);
}
