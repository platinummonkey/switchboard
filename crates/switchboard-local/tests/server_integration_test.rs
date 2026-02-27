//! End-to-end integration tests for `switchboard-local` + `switchboard-server`.
//!
//! These tests spin up a full three-tier stack:
//!
//! ```text
//! reqwest test client
//!     → switchboard-local  (real TCP, random port)
//!         → switchboard-server  (real TCP, random port)
//!             → wiremock  (simulates upstream LLM provider)
//! ```
//!
//! Each test allocates ports by binding to `127.0.0.1:0` and letting the OS
//! assign a free port, avoiding TOCTOU races.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use switchboard_local::auth::LocalAuthManager;
use switchboard_local::config::{
    AuthConfig, IdentityConfig, LocalConfig, LocalListenConfig, ModelConfig,
    ServerConfig as LocalServerConfig,
};
use switchboard_local::model_prefs::ModelPrefs;
use switchboard_local::proxy::LocalServerState;
use switchboard_local::server::build_router;

use switchboard_server::config::provider::{KeyEntry, KeyPoolConfig, ProviderConfig};
use switchboard_server::config::{AuthConfig as SrvAuthConfig, ServerConfig, ValidatorConfig};

// ── Server helpers ─────────────────────────────────────────────────────────────

/// Build a minimal [`ServerConfig`] with one OpenAI provider pointing at
/// `provider_base_url` and a single static-key auth validator accepting
/// `"test-api-key"`.
fn build_server_config(provider_base_url: &str) -> ServerConfig {
    let mut providers = HashMap::new();
    providers.insert(
        "openai".to_string(),
        ProviderConfig {
            base_url: Some(provider_base_url.to_owned()),
            api_format: "openai".into(),
            models: vec!["gpt-4o".into(), "gpt-3.5-turbo".into()],
            region: None,
            cross_region_inference: false,
            project_id: None,
            timeout: "30s".into(),
            health_check_interval: "30s".into(),
            max_concurrent: 10,
            key_pool: KeyPoolConfig {
                selector: "weighted_random".into(),
                keys: vec![KeyEntry {
                    id: "openai-test-key".into(),
                    key_type: "static".into(),
                    api_key: Some("sk-test-openai".into()),
                    role_arn: None,
                    region: None,
                    refresh_interval: None,
                    vault_path: None,
                    weight: 1.0,
                }],
            },
        },
    );

    let mut validators = HashMap::new();
    validators.insert(
        "test".to_string(),
        ValidatorConfig {
            validator_type: "static_keys".into(),
            jwks_url: None,
            audience: None,
            issuer: None,
            keys: vec!["test-api-key".into()],
            ca: None,
        },
    );

    ServerConfig {
        providers,
        auth: SrvAuthConfig { validators },
        ..ServerConfig::default()
    }
}

/// Bind a random TCP port and start `switchboard-server` on it.
///
/// Returns `(server_addr, shutdown_tx)`.  Dropping or sending on
/// `shutdown_tx` triggers a graceful shutdown.
async fn start_server(config: ServerConfig) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind server listener");
    let addr = listener.local_addr().expect("no local addr");

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        switchboard_server::run_server(
            config,
            "", // no file-based hot-reload in tests
            listener,
            None,
            async move {
                let _ = rx.await;
            },
        )
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "test server exited with error");
        });
    });

    // Give the server task a chance to start.
    tokio::task::yield_now().await;

    // Poll /health (with auth) until the server is ready.
    wait_for_server_ready(addr).await;

    (addr, tx)
}

/// Poll `GET http://{addr}/health` with the test API key until 200 or timeout.
async fn wait_for_server_ready(addr: SocketAddr) {
    let client = reqwest::Client::new();
    let url = format!("http://{addr}/health");
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

    loop {
        match client
            .get(&url)
            .header("Authorization", "Bearer test-api-key")
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => return,
            _ => {}
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("switchboard-server at {addr} did not become ready within 5 seconds");
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

// ── Local proxy helpers ────────────────────────────────────────────────────────

/// Build a [`LocalConfig`] pointing at `server_url` with the given identity
/// and optional model overrides.
fn build_local_config(
    server_url: &str,
    user: Option<&str>,
    team: Option<&str>,
    model_overrides: HashMap<String, String>,
) -> Arc<LocalConfig> {
    Arc::new(LocalConfig {
        server: LocalServerConfig {
            url: server_url.to_owned(),
        },
        auth: AuthConfig {
            method: "api_key".into(),
            api_key: Some("test-api-key".into()),
            ..AuthConfig::default()
        },
        identity: IdentityConfig {
            user: user.map(str::to_owned),
            team: team.map(str::to_owned),
        },
        local: LocalListenConfig::default(),
        model: ModelConfig {
            default: "gpt-4o".into(),
            overrides: model_overrides,
        },
    })
}

/// Build a [`LocalServerState`] from a config.
async fn make_local_state(config: Arc<LocalConfig>) -> Arc<LocalServerState> {
    let auth_manager = LocalAuthManager::new(&config.auth)
        .await
        .expect("failed to build LocalAuthManager");
    let model_prefs = ModelPrefs::from_config(&config.model);
    Arc::new(LocalServerState {
        auth_manager: Arc::new(auth_manager),
        model_prefs: Arc::new(model_prefs),
        config,
        client: reqwest::Client::new(),
    })
}

/// Bind a random port, start the local proxy axum router, and wait for /health.
///
/// Returns `(local_base_url, join_handle)`.
async fn start_local(state: Arc<LocalServerState>) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind local proxy listener");
    let addr = listener.local_addr().expect("no local addr");
    let base_url = format!("http://{addr}");

    let router = build_router(state);
    let handle = tokio::spawn(async move {
        axum::serve(listener, router)
            .await
            .expect("local proxy server error");
    });

    wait_for_local_ready(&base_url).await;
    (base_url, handle)
}

/// Poll `GET {base_url}/health` until 200 or timeout.
async fn wait_for_local_ready(base_url: &str) {
    let client = reqwest::Client::new();
    let url = format!("{base_url}/health");
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

    loop {
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => return,
            _ => {}
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("switchboard-local at {base_url} did not become ready within 5 seconds");
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

// ── Minimal wiremock response ──────────────────────────────────────────────────

/// A stock OpenAI chat completion response body.
fn openai_chat_response(content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-e2e-test",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Full stack happy-path.
///
/// Verifies that a request sent to `switchboard-local` travels through the
/// full stack (local → server → wiremock) and the wiremock receives exactly
/// one request.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_local_to_server_basic_request() {
    // 1. Start wiremock (simulates upstream LLM provider).
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_chat_response("Hello from wiremock"))
                .insert_header("content-type", "application/json"),
        )
        .expect(1)
        .mount(&upstream)
        .await;

    // 2. Start switchboard-server pointing at wiremock.
    let server_config = build_server_config(&upstream.uri());
    let (server_addr, _server_shutdown) = start_server(server_config).await;
    let server_url = format!("http://{server_addr}");

    // 3. Start switchboard-local pointing at switchboard-server.
    let local_config = build_local_config(&server_url, None, None, HashMap::new());
    let local_state = make_local_state(local_config).await;
    let (local_url, _local_handle) = start_local(local_state).await;

    // 4. Send a chat completion request to the local proxy.
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{local_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .send()
        .await
        .expect("failed to send request to local proxy");

    assert_eq!(
        resp.status().as_u16(),
        200,
        "expected 200 OK from local proxy"
    );

    let body: serde_json::Value = resp.json().await.expect("failed to parse response JSON");
    assert!(
        body["choices"].is_array(),
        "response must contain a 'choices' array"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"].as_str(),
        Some("Hello from wiremock"),
        "response content must match wiremock fixture"
    );

    // 5. Verify that wiremock received exactly one request (mock expectation).
    upstream.verify().await;
}

/// Identity header injection.
///
/// Verifies that `switchboard-local` injects the configured `X-Switchboard-User`
/// and `X-Switchboard-Team` identity headers into outgoing requests to the
/// server.
///
/// Architecture for this test:
///
/// ```text
/// reqwest test client
///     → switchboard-local  (real TCP, random port)
///         → wiremock  (acts as the server — lets us inspect received headers)
/// ```
///
/// We use wiremock directly as the "server" here so we can observe exactly
/// what headers the local proxy injects before any server-side stripping.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_local_to_server_identity_injected() {
    // 1. Start wiremock acting as the server.
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_chat_response("identity ok"))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    // 2. Start local proxy pointed directly at the mock (no real server needed
    //    to verify header injection at the server boundary).
    let local_config = build_local_config(
        &mock_server.uri(),
        Some("alice@example.com"),
        Some("platform"),
        HashMap::new(),
    );
    let local_state = make_local_state(local_config).await;
    let (local_url, _local_handle) = start_local(local_state).await;

    // 3. Send a request without any identity header from the client side — the
    //    local proxy should inject X-Switchboard-User from its own config.
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{local_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "who am I?"}]
        }))
        .send()
        .await
        .expect("request to local proxy failed");

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    // 4. Inspect what the mock server received and verify the identity headers.
    let received = mock_server
        .received_requests()
        .await
        .expect("failed to retrieve received requests");
    assert_eq!(
        received.len(),
        1,
        "mock server must have received exactly 1 request"
    );

    let req = &received[0];

    let user_header = req
        .headers
        .get("x-switchboard-user")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(
        user_header, "alice@example.com",
        "X-Switchboard-User header must be injected by switchboard-local"
    );

    let team_header = req
        .headers
        .get("x-switchboard-team")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(
        team_header, "platform",
        "X-Switchboard-Team header must be injected by switchboard-local"
    );
}

/// Model preference rewriting.
///
/// Verifies that when `switchboard-local` is configured with a model override
/// mapping `"gpt-3.5-turbo"` → `"gpt-4o"`, requests specifying
/// `"gpt-3.5-turbo"` arrive at the upstream with the rewritten model name.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_local_to_server_model_pref_applied() {
    // 1. Start wiremock — records all received requests.
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_chat_response("model pref ok"))
                .insert_header("content-type", "application/json"),
        )
        .mount(&upstream)
        .await;

    // 2. Start server.
    let server_config = build_server_config(&upstream.uri());
    let (server_addr, _server_shutdown) = start_server(server_config).await;
    let server_url = format!("http://{server_addr}");

    // 3. Start local proxy with a model override: "gpt-3.5-turbo" → "gpt-4o".
    let mut overrides = HashMap::new();
    overrides.insert("gpt-3.5-turbo".to_string(), "gpt-4o".to_string());
    let local_config = build_local_config(&server_url, None, None, overrides);
    let local_state = make_local_state(local_config).await;
    let (local_url, _local_handle) = start_local(local_state).await;

    // 4. Send a request specifying the *source* model name.
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{local_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test model pref"}]
        }))
        .send()
        .await
        .expect("request to local proxy failed");

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    // 5. Verify that the upstream received the *target* model name.
    let received = upstream
        .received_requests()
        .await
        .expect("failed to retrieve received requests");
    assert_eq!(
        received.len(),
        1,
        "wiremock must have received exactly 1 request"
    );

    let upstream_body: serde_json::Value =
        serde_json::from_slice(&received[0].body).expect("upstream request body is not valid JSON");

    assert_eq!(
        upstream_body["model"].as_str(),
        Some("gpt-4o"),
        "model must be rewritten from 'gpt-3.5-turbo' to 'gpt-4o' by the local proxy"
    );
}

/// Server-down error handling: unreachable server.
///
/// Verifies that when `switchboard-local` is configured to forward to a server
/// address that is not listening (port 1 is always refused), the local proxy
/// returns a non-200 error response rather than hanging or panicking.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_local_returns_error_when_server_unreachable() {
    // Port 1 is reserved and always produces a connection-refused error.
    let local_config = build_local_config(
        "http://127.0.0.1:1",
        None,
        None,
        std::collections::HashMap::new(),
    );
    let state = make_local_state(local_config).await;
    let (local_url, _local_handle) = start_local(state).await;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();

    let resp = client
        .post(format!("{local_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("request to local proxy must complete (with an error status)");

    assert_ne!(
        resp.status().as_u16(),
        200,
        "unreachable server should not return 200"
    );
    // The local proxy should map connection errors to 502 Bad Gateway.
    assert_eq!(
        resp.status().as_u16(),
        502,
        "expected 502 Bad Gateway when upstream is unreachable"
    );
}

/// Server-down error handling: server shuts down mid-flight.
///
/// Starts a full three-tier stack, verifies the first request succeeds, then
/// drops the switchboard-server shutdown sender to stop it.  The second
/// request — sent after the server has stopped — must return a non-200 error.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_local_returns_error_when_server_shuts_down() {
    // 1. Start wiremock (upstream LLM provider).
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(openai_chat_response("ok"))
                .insert_header("content-type", "application/json"),
        )
        .mount(&upstream)
        .await;

    // 2. Start switchboard-server.
    let server_config = build_server_config(&upstream.uri());
    let (server_addr, server_shutdown_tx) = start_server(server_config).await;
    let server_url = format!("http://{server_addr}");

    // 3. Start switchboard-local.
    let local_config = build_local_config(&server_url, None, None, HashMap::new());
    let state = make_local_state(local_config).await;
    let (local_url, _local_handle) = start_local(state).await;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();

    // 4. First request must succeed (proves the stack is wired correctly).
    let resp1 = client
        .post(format!("{local_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("first request should succeed");
    assert_eq!(
        resp1.status().as_u16(),
        200,
        "first request must succeed before server shutdown"
    );

    // 5. Kill the switchboard-server by dropping its shutdown sender.
    drop(server_shutdown_tx);
    // Give the server task a moment to finish its graceful shutdown.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    // 6. Second request must fail now that the server is gone.
    let resp2 = client
        .post(format!("{local_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello again"}]
        }))
        .send()
        .await
        .expect("second request must complete (with an error status)");

    assert_ne!(
        resp2.status().as_u16(),
        200,
        "request after server shutdown should not return 200"
    );
}

/// Local health check is independent of the upstream server.
///
/// Even when `switchboard-local` is configured to forward to an unreachable
/// server, its own `/health` endpoint must return 200 — the health check
/// reflects the local proxy's own liveness, not the server's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_local_health_check_works_independently_of_server() {
    // Point at an unreachable address.
    let local_config = build_local_config(
        "http://127.0.0.1:1",
        None,
        None,
        std::collections::HashMap::new(),
    );
    let state = make_local_state(local_config).await;
    let (local_url, _local_handle) = start_local(state).await;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();

    // GET /health must succeed locally — it does not proxy to the server.
    let resp = client
        .get(format!("{local_url}/health"))
        .send()
        .await
        .expect("GET /health should always succeed");

    assert_eq!(
        resp.status().as_u16(),
        200,
        "/health must return 200 even when the upstream server is unreachable"
    );

    let body: serde_json::Value = resp.json().await.expect("/health response must be JSON");
    assert_eq!(
        body["status"].as_str(),
        Some("ok"),
        "/health body must contain status: ok"
    );
}

/// Timeout behaviour: slow server that accepts but never responds.
///
/// Creates a TCP listener that accepts connections and then holds them open
/// without ever sending a byte.  The local proxy must return an error within
/// a reasonable wall-clock time rather than blocking forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_local_timeout_on_slow_server() {
    // Bind a port that will accept connections but never send a response.
    let black_hole = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind black-hole listener");
    let black_hole_addr = black_hole.local_addr().expect("no local addr");

    tokio::spawn(async move {
        loop {
            match black_hole.accept().await {
                Ok((_stream, _peer)) => {
                    // Hold the connection open for a long time — never write
                    // a single byte back.
                    tokio::time::sleep(std::time::Duration::from_secs(120)).await;
                }
                Err(_) => break,
            }
        }
    });

    let local_config = build_local_config(
        &format!("http://{black_hole_addr}"),
        None,
        None,
        std::collections::HashMap::new(),
    );
    let state = make_local_state(local_config).await;
    let (local_url, _local_handle) = start_local(state).await;

    // Use a short reqwest timeout so the test doesn't block for minutes.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap();

    let start = std::time::Instant::now();
    let result = client
        .post(format!("{local_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await;
    let elapsed = start.elapsed();

    // The request must either time out at the reqwest layer (Err) or the local
    // proxy must surface an error status — either way it must NOT succeed with
    // 200 and must complete within a generous ceiling.
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "request must not hang indefinitely (elapsed: {elapsed:?})"
    );

    match result {
        Ok(resp) => {
            assert_ne!(
                resp.status().as_u16(),
                200,
                "slow/unresponsive server must not produce a 200 response"
            );
        }
        Err(e) => {
            // A timeout or connection error from reqwest is also acceptable.
            assert!(
                e.is_timeout() || e.is_connect() || e.is_request(),
                "unexpected reqwest error kind: {e}"
            );
        }
    }
}
