//! Integration tests for the `switchboard-local` proxy.
//!
//! Each test spins up a [`wiremock::MockServer`] to simulate `switchboard-server`,
//! builds a [`LocalServer`] (or just the router) pointed at that mock, and makes
//! real HTTP requests to verify correct forwarding behaviour.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Router;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── helpers ───────────────────────────────────────────────────────────────────

use switchboard_local::auth::LocalAuthManager;
use switchboard_local::config::{
    AuthConfig, IdentityConfig, LocalConfig, LocalListenConfig, ModelConfig, ServerConfig,
};
use switchboard_local::model_prefs::ModelPrefs;
use switchboard_local::proxy::LocalServerState;
use switchboard_local::server::build_router;

/// Build a minimal [`LocalConfig`] that points to the given `wiremock` URL.
fn make_config(server_url: &str) -> Arc<LocalConfig> {
    Arc::new(LocalConfig {
        server: ServerConfig {
            url: server_url.to_owned(),
        },
        auth: AuthConfig {
            method: "api_key".into(),
            api_key: Some("sk-test-proxy-key".into()),
            ..AuthConfig::default()
        },
        identity: IdentityConfig {
            user: Some("test-user@example.com".into()),
            team: Some("platform".into()),
        },
        local: LocalListenConfig::default(),
        model: ModelConfig {
            default: "claude-sonnet-4-20250514".into(),
            overrides: {
                let mut m = HashMap::new();
                m.insert("gpt-4".into(), "claude-sonnet-4-20250514".into());
                m
            },
        },
    })
}

/// Bind a `tokio` listener on a random port, serve `router`, and return the
/// base URL for the bound address.
async fn serve_router(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{}", addr);
    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base_url, handle)
}

/// Build a [`LocalServerState`] pointing at `mock_server`.
async fn make_state(mock_server: &MockServer) -> Arc<LocalServerState> {
    let config = make_config(&mock_server.uri());
    let auth_manager = LocalAuthManager::new(&config.auth).await.unwrap();
    let model_prefs = ModelPrefs::from_config(&config.model);
    let client = reqwest::Client::new();
    Arc::new(LocalServerState {
        config,
        auth_manager: Arc::new(auth_manager),
        model_prefs: Arc::new(model_prefs),
        client,
    })
}

// ── test_local_proxy_openai_non_streaming ─────────────────────────────────────

#[tokio::test]
async fn test_local_proxy_openai_non_streaming() {
    let mock_server = MockServer::start().await;

    let openai_response = serde_json::json!({
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(&openai_response)
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let state = make_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve_router(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", base_url))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["id"].as_str(), Some("chatcmpl-abc123"));
    assert_eq!(
        body["choices"][0]["message"]["content"].as_str(),
        Some("Hello!")
    );
}

// ── test_local_proxy_anthropic_non_streaming ──────────────────────────────────

#[tokio::test]
async fn test_local_proxy_anthropic_non_streaming() {
    let mock_server = MockServer::start().await;

    let anthropic_response = serde_json::json!({
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello from Anthropic!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });

    Mock::given(method("POST"))
        .and(path("/api/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(&anthropic_response)
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let state = make_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve_router(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/api/v1/messages", base_url))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"].as_str(), Some("message"));
    assert_eq!(
        body["content"][0]["text"].as_str(),
        Some("Hello from Anthropic!")
    );
}

// ── test_local_proxy_injects_auth_header ──────────────────────────────────────

#[tokio::test]
async fn test_local_proxy_injects_auth_header() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-test-proxy-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"id": "ok"}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let state = make_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve_router(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", base_url))
        .json(&serde_json::json!({"model": "claude-sonnet-4-20250514", "messages": []}))
        .send()
        .await
        .unwrap();

    // wiremock only matches when the auth header IS present, so 200 = header was injected
    assert_eq!(resp.status().as_u16(), 200);
}

// ── test_local_proxy_injects_user_header ─────────────────────────────────────

#[tokio::test]
async fn test_local_proxy_injects_user_header() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("x-switchboard-user", "test-user@example.com"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"id": "user-ok"}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let state = make_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve_router(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", base_url))
        .json(&serde_json::json!({"model": "claude-sonnet-4-20250514", "messages": []}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
}

// ── test_local_proxy_model_override ──────────────────────────────────────────

#[tokio::test]
async fn test_local_proxy_model_override() {
    let mock_server = MockServer::start().await;

    // We expect the mock server to receive "claude-sonnet-4-20250514", not "gpt-4".
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"id": "override-ok"}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let state = make_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve_router(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", base_url))
        // Client sends "gpt-4" — proxy should rewrite to "claude-sonnet-4-20250514"
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);

    // Verify wiremock actually received the rewritten model by inspecting
    // the recorded requests.
    let received = mock_server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(
        body["model"].as_str(),
        Some("claude-sonnet-4-20250514"),
        "Expected model to be rewritten from gpt-4 to claude-sonnet-4-20250514"
    );
}

// ── test_local_proxy_health_endpoint ─────────────────────────────────────────

#[tokio::test]
async fn test_local_proxy_health_endpoint() {
    let mock_server = MockServer::start().await;

    let state = make_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve_router(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"].as_str(), Some("ok"));
    assert!(body["server"].as_str().is_some());
    assert_eq!(body["auth"].as_str(), Some("api_key"));
}

// ── test_model_prefs_override ─────────────────────────────────────────────────

#[test]
fn test_model_prefs_override() {
    let mut overrides = HashMap::new();
    overrides.insert("gpt-4".into(), "claude-sonnet-4-20250514".into());
    let prefs = ModelPrefs::from_config(&ModelConfig {
        default: "claude-sonnet-4-20250514".into(),
        overrides,
    });

    assert_eq!(
        prefs.override_model("gpt-4"),
        Some("claude-sonnet-4-20250514")
    );
    assert_eq!(prefs.override_model("unknown-model"), None);
}

// ── test_model_prefs_default ──────────────────────────────────────────────────

#[test]
fn test_model_prefs_default() {
    let prefs = ModelPrefs::from_config(&ModelConfig {
        default: "claude-opus-4-20250514".into(),
        overrides: HashMap::new(),
    });

    assert_eq!(prefs.default_model(), "claude-opus-4-20250514");
}

// ── test_model_prefs_apply_to_body ────────────────────────────────────────────

#[test]
fn test_model_prefs_apply_to_body() {
    let mut overrides = HashMap::new();
    overrides.insert("gpt-4".into(), "claude-sonnet-4-20250514".into());
    let prefs = ModelPrefs::from_config(&ModelConfig {
        default: "claude-sonnet-4-20250514".into(),
        overrides,
    });

    // Case 1: override exists — model field is rewritten.
    let mut body = serde_json::json!({"model": "gpt-4", "messages": []});
    prefs.apply_to_body(&mut body);
    assert_eq!(body["model"].as_str(), Some("claude-sonnet-4-20250514"));

    // Case 2: no override — model field unchanged.
    let mut body2 = serde_json::json!({"model": "some-other-model", "messages": []});
    prefs.apply_to_body(&mut body2);
    assert_eq!(body2["model"].as_str(), Some("some-other-model"));

    // Case 3: no model field — body unchanged.
    let mut body3 = serde_json::json!({"messages": []});
    let original3 = body3.clone();
    prefs.apply_to_body(&mut body3);
    assert_eq!(body3, original3);
}
