//! End-to-end integration tests for `switchboard-local`.
//!
//! Each test spins up a [`wiremock::MockServer`] to simulate
//! `switchboard-server`, builds a [`LocalServerState`], mounts the axum router,
//! and exercises the proxy + header-injection logic via real HTTP requests.

use std::collections::HashMap;
use std::sync::Arc;

use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use switchboard_local::auth::LocalAuthManager;
use switchboard_local::config::{
    AuthConfig, IdentityConfig, LocalConfig, LocalListenConfig, ModelConfig, ServerConfig,
};
use switchboard_local::model_prefs::ModelPrefs;
use switchboard_local::proxy::LocalServerState;
use switchboard_local::server::build_router;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a minimal `LocalConfig` pointing at `server_url` with the given API
/// key, user, and team.
fn make_config_with_identity(
    server_url: &str,
    api_key: &str,
    user: &str,
    team: &str,
) -> Arc<LocalConfig> {
    Arc::new(LocalConfig {
        server: ServerConfig {
            url: server_url.to_owned(),
        },
        auth: AuthConfig {
            method: "api_key".into(),
            api_key: Some(api_key.to_owned()),
            ..AuthConfig::default()
        },
        identity: IdentityConfig {
            user: Some(user.to_owned()),
            team: Some(team.to_owned()),
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

/// Build a `LocalServerState` from a config.
async fn make_state_from_config(config: Arc<LocalConfig>) -> Arc<LocalServerState> {
    let auth_manager = LocalAuthManager::new(&config.auth).await.unwrap();
    let model_prefs = ModelPrefs::from_config(&config.model);
    Arc::new(LocalServerState {
        auth_manager: Arc::new(auth_manager),
        model_prefs: Arc::new(model_prefs),
        config,
        client: reqwest::Client::new(),
    })
}

/// Convenience: build state pointed at `mock_server` with default test credentials.
async fn make_test_state(mock_server: &MockServer) -> Arc<LocalServerState> {
    let config = make_config_with_identity(
        &mock_server.uri(),
        "sk-local-test",
        "test@example.com",
        "testing",
    );
    make_state_from_config(config).await
}

/// Bind on a random port, serve `router`, return `(base_url, join_handle)`.
async fn serve(router: axum::Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");
    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base_url, handle)
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// POST to `/v1/chat/completions` round-trips and returns the upstream JSON
/// with a `choices` array.
#[tokio::test]
async fn test_local_proxy_openai_roundtrip() {
    let mock_server = MockServer::start().await;

    let openai_response = serde_json::json!({
        "id": "chatcmpl-roundtrip",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi there!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
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

    let state = make_test_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["choices"].is_array(),
        "response must have a 'choices' array"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"].as_str(),
        Some("Hi there!")
    );
}

/// POST to `/api/v1/messages` round-trips and returns the upstream Anthropic
/// JSON (type = "message").
#[tokio::test]
async fn test_local_proxy_anthropic_roundtrip() {
    let mock_server = MockServer::start().await;

    let anthropic_response = serde_json::json!({
        "id": "msg_roundtrip",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Greetings!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2}
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

    let state = make_test_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/api/v1/messages"))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"].as_str(), Some("message"));
    assert_eq!(body["content"][0]["text"].as_str(), Some("Greetings!"));
}

/// The local proxy must inject `Authorization: Bearer sk-local-test`.
/// Wiremock only matches when the header is present, so a 200 confirms injection.
#[tokio::test]
async fn test_local_proxy_injects_auth_header() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-local-test"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"id": "auth-ok"}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let state = make_test_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "claude-sonnet-4-20250514", "messages": []}))
        .send()
        .await
        .unwrap();

    // 200 confirms the mock matched, meaning the auth header was injected.
    assert_eq!(resp.status().as_u16(), 200);
}

/// The local proxy must inject `X-Switchboard-User: test@example.com`.
#[tokio::test]
async fn test_local_proxy_injects_user_header() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("x-switchboard-user", "test@example.com"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"id": "user-ok"}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let state = make_test_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "claude-sonnet-4-20250514", "messages": []}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
}

/// The local proxy must inject `X-Switchboard-Team: testing`.
#[tokio::test]
async fn test_local_proxy_injects_team_header() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("x-switchboard-team", "testing"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"id": "team-ok"}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let state = make_test_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "claude-sonnet-4-20250514", "messages": []}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
}

/// When `"gpt-4"` is mapped to `"claude-sonnet-4-20250514"`, the upstream must
/// receive the overridden model name in the request body.
#[tokio::test]
async fn test_local_proxy_model_override_applied() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"id": "override-ok"}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let state = make_test_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);

    // Inspect recorded requests to verify the override was applied.
    let recorded = mock_server.received_requests().await.unwrap();
    assert_eq!(
        recorded.len(),
        1,
        "mock should have received exactly one request"
    );
    let upstream_body: serde_json::Value = serde_json::from_slice(&recorded[0].body).unwrap();
    assert_eq!(
        upstream_body["model"].as_str(),
        Some("claude-sonnet-4-20250514"),
        "model field must be rewritten from 'gpt-4' to 'claude-sonnet-4-20250514'"
    );
}

/// When the request body contains no `model` field, the proxy injects the
/// configured default model via the `X-Switchboard-Model` header and also
/// refrains from injecting into the body (body pass-through is used here since
/// the model field is absent).
///
/// We verify the upstream receives a request that the mock accepts (200 OK).
#[tokio::test]
async fn test_model_prefs_default_injected_when_no_model() {
    let mock_server = MockServer::start().await;

    // The mock accepts any POST — we inspect the recorded body afterwards.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"id": "default-model-ok"}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let config = Arc::new(LocalConfig {
        server: ServerConfig {
            url: mock_server.uri(),
        },
        auth: AuthConfig {
            method: "api_key".into(),
            api_key: Some("sk-local-test".into()),
            ..AuthConfig::default()
        },
        identity: IdentityConfig {
            user: Some("test@example.com".into()),
            team: Some("testing".into()),
        },
        local: LocalListenConfig::default(),
        model: ModelConfig {
            default: "claude-sonnet-4-20250514".into(),
            overrides: HashMap::new(),
        },
    });
    let state = make_state_from_config(config).await;
    let router = build_router(state);
    let (base_url, _handle) = serve(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        // No "model" field — proxy should inject X-Switchboard-Model header.
        .json(&serde_json::json!({
            "messages": [{"role": "user", "content": "ping"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);

    // The X-Switchboard-Model header must have been forwarded.
    let recorded = mock_server.received_requests().await.unwrap();
    assert_eq!(recorded.len(), 1);
    let sw_model = recorded[0]
        .headers
        .get("x-switchboard-model")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(
        sw_model, "claude-sonnet-4-20250514",
        "X-Switchboard-Model should carry the configured default model"
    );
}

/// GET `/health` must return 200 with a JSON body that has `status = "ok"`.
#[tokio::test]
async fn test_local_health_endpoint() {
    let mock_server = MockServer::start().await;
    let state = make_test_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["status"].as_str(),
        Some("ok"),
        "health status must be 'ok'"
    );
    assert!(
        body["server"].as_str().is_some(),
        "health response must include 'server' field"
    );
    assert_eq!(
        body["auth"].as_str(),
        Some("api_key"),
        "health response must reflect auth method"
    );
}

/// When the upstream returns 500, the local proxy must propagate a 5xx status
/// code to the caller (either 500 forwarded as-is, or 502 Bad Gateway).
#[tokio::test]
async fn test_local_proxy_upstream_error_propagated() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let state = make_test_state(&mock_server).await;
    let router = build_router(state);
    let (base_url, _handle) = serve(router).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "error test"}]
        }))
        .send()
        .await
        .unwrap();

    let status = resp.status().as_u16();
    assert!(
        status == 500 || status == 502,
        "upstream 500 must yield 500 or 502, got {status}"
    );
}
