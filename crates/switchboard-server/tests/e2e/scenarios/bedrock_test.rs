//! E2E scenario: Bedrock Converse API.
//!
//! Tests cover:
//! - Basic chat via the Bedrock Converse endpoint
//! - Proxy converts Bedrock response format to OpenAI format
//! - Upstream errors are surfaced to the caller
//! - Mock server receives exactly the expected number of requests
//! - Streaming requests hit the `/converse-stream` path (not `/converse`)
//! - Non-streaming and streaming requests use distinct upstream endpoints
//! - Cross-region inference prepends the correct regional prefix to the model
//!   ID in the upstream URL path

use std::collections::HashMap;

use serde_json::json;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

use switchboard_server::config::admin::AdminConfig;
use switchboard_server::config::guardrails::GuardrailsConfig;
use switchboard_server::config::model_selection::ModelSelectionConfig;
use switchboard_server::config::provider::{KeyEntry, KeyPoolConfig, ProviderConfig};
use switchboard_server::config::rate_limit::RateLimitConfig;
use switchboard_server::config::routing::RoutingConfig;
use switchboard_server::config::{AuthConfig, ServerConfig, ValidatorConfig};

use crate::assertions::{assert_openai_chat_response, assert_received_n};
use crate::harness::TestHarnessBuilder;
use crate::mocks::bedrock;

// ── Local mock helpers ────────────────────────────────────────────────────────

/// Wiremock mock for the Bedrock streaming Converse endpoint.
///
/// Matches `POST /model/{model_id}/converse-stream` and returns the given
/// `content` string as a raw byte body with the AWS event-stream content-type.
///
/// NOTE: Real Bedrock streaming uses AWS event-stream binary framing.  In the
/// test environment the proxy forwards these bytes verbatim — it does NOT
/// attempt to parse or re-frame them.  Tests therefore only assert on the HTTP
/// status code and that a non-empty body was received, rather than trying to
/// decode the binary framing.
fn mock_converse_stream_ok(content: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path_regex(r"^/model/.+/converse-stream$"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/vnd.amazon.eventstream")
                .set_body_string(content.to_string()),
        )
}

/// 1. A basic chat request routed to Bedrock returns 200 and the correct content.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_bedrock_converse_basic_chat() {
    let harness = TestHarnessBuilder::new().with_bedrock().build().await;

    let mock_server = harness
        .mocks
        .bedrock
        .as_ref()
        .expect("bedrock mock must be present");
    bedrock::mock_converse_ok("Paris").mount(mock_server).await;

    let body = json!({
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "Capital of France?"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "expected 200 OK from Bedrock proxy"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["choices"][0]["message"]["content"], "Paris",
        "content should be 'Paris' as returned by the Bedrock mock"
    );
}

/// 2. The proxy converts the Bedrock Converse response into the OpenAI chat
///    completion format, including `choices` and `usage` with `prompt_tokens`
///    and `completion_tokens`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_bedrock_response_converted_to_openai_format() {
    let harness = TestHarnessBuilder::new().with_bedrock().build().await;

    let mock_server = harness
        .mocks
        .bedrock
        .as_ref()
        .expect("bedrock mock must be present");
    bedrock::mock_converse_ok("Converted response")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "Say something."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");

    assert_openai_chat_response(&json, "Converted response");

    // The `choices` array must be present and non-empty.
    let choices = json["choices"]
        .as_array()
        .expect("choices must be an array");
    assert!(!choices.is_empty(), "choices array must not be empty");

    // `usage` must have `prompt_tokens` and `completion_tokens`.
    let usage = &json["usage"];
    assert!(
        usage["prompt_tokens"].as_u64().unwrap_or(0) > 0,
        "prompt_tokens must be non-zero"
    );
    assert!(
        usage["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "completion_tokens must be non-zero"
    );
}

/// 3. When the Bedrock upstream responds with an error (e.g., ThrottlingException),
///    the proxy returns a non-200 status to the caller.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_bedrock_upstream_error_returned() {
    let harness = TestHarnessBuilder::new().with_bedrock().build().await;

    let mock_server = harness
        .mocks
        .bedrock
        .as_ref()
        .expect("bedrock mock must be present");
    bedrock::mock_converse_error(429, "ThrottlingException", "Rate limit exceeded")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "Capital of France?"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_ne!(
        resp.status().as_u16(),
        200,
        "a Bedrock upstream error must not be returned as 200"
    );
}

/// 4. Exactly one request reaches the Bedrock mock server for a single chat call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_bedrock_mock_receives_request() {
    let harness = TestHarnessBuilder::new().with_bedrock().build().await;

    let mock_server = harness
        .mocks
        .bedrock
        .as_ref()
        .expect("bedrock mock must be present");
    bedrock::mock_converse_ok("one request")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "Ping?"}]
    });

    harness.client.chat_completions(body).await;

    assert_received_n(mock_server, 1).await;
}

// ── Streaming tests ───────────────────────────────────────────────────────────

/// 5. A streaming request to the Bedrock provider returns HTTP 200 and a
///    non-empty response body.
///
/// Bedrock streaming uses AWS event-stream binary framing, which the proxy
/// forwards verbatim without re-encoding as text SSE.  The test therefore only
/// checks that the status is 200 and that at least some bytes were received —
/// it does not attempt to parse the binary payload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_bedrock_streaming_returns_response() {
    let harness = TestHarnessBuilder::new().with_bedrock().build().await;

    let mock_server = harness
        .mocks
        .bedrock
        .as_ref()
        .expect("bedrock mock must be present");
    mock_converse_stream_ok("streamed bedrock content")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "Stream something!"}]
    });

    let resp = harness.client.chat_completions_stream(body).await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "expected 200 OK from Bedrock streaming proxy"
    );

    // Collect the response body and verify at least some bytes arrived.
    // We do NOT attempt to parse the binary event-stream framing here.
    let bytes = resp.bytes().await.expect("failed to read streaming body");
    assert!(
        !bytes.is_empty(),
        "expected a non-empty body from the Bedrock streaming mock"
    );
}

/// 6. A streaming request reaches the Bedrock mock via the `/converse-stream`
///    path, not the non-streaming `/converse` path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_bedrock_streaming_mock_receives_converse_stream_path() {
    let harness = TestHarnessBuilder::new().with_bedrock().build().await;

    let mock_server = harness
        .mocks
        .bedrock
        .as_ref()
        .expect("bedrock mock must be present");
    mock_converse_stream_ok("path check")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "Which endpoint?"}]
    });

    harness.client.chat_completions_stream(body).await;

    // Inspect the raw request that arrived at the mock to verify the path.
    let reqs = mock_server
        .received_requests()
        .await
        .expect("failed to fetch received requests");

    assert_eq!(reqs.len(), 1, "expected exactly one request to the mock");

    let path = reqs[0].url.path();
    assert!(
        path.ends_with("/converse-stream"),
        "streaming request must target /converse-stream, got: {}",
        path
    );
    assert!(
        !path.ends_with("/converse"),
        "streaming request must NOT target the non-streaming /converse endpoint"
    );
}

/// 7. Non-streaming and streaming requests use different upstream endpoints:
///    - Non-streaming → `/model/{id}/converse`
///    - Streaming     → `/model/{id}/converse-stream`
///
/// Both mocks are mounted on the same wiremock server.  Wiremock dispatches
/// each incoming request to the first matching mock (path patterns differ, so
/// there is no ambiguity).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_bedrock_non_streaming_vs_streaming_different_paths() {
    let harness = TestHarnessBuilder::new().with_bedrock().build().await;

    let mock_server = harness
        .mocks
        .bedrock
        .as_ref()
        .expect("bedrock mock must be present");

    // Mount both mocks — different path regexes ensure no ambiguity.
    bedrock::mock_converse_ok("non-streaming response")
        .mount(mock_server)
        .await;
    mock_converse_stream_ok("streaming response")
        .mount(mock_server)
        .await;

    let model_body = json!({
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    // ── Non-streaming request ────────────────────────────────────────────────
    harness.client.chat_completions(model_body.clone()).await;

    // Exactly one request should have arrived so far.
    assert_received_n(mock_server, 1).await;

    // Verify it hit the non-streaming path.
    let reqs = mock_server
        .received_requests()
        .await
        .expect("failed to fetch received requests");
    let non_stream_path = reqs[0].url.path();
    assert!(
        non_stream_path.ends_with("/converse"),
        "non-streaming request must target /converse, got: {}",
        non_stream_path
    );
    assert!(
        !non_stream_path.ends_with("/converse-stream"),
        "non-streaming request must NOT target /converse-stream"
    );

    // ── Streaming request ────────────────────────────────────────────────────
    harness.client.chat_completions_stream(model_body).await;

    // Now two requests total should have arrived.
    assert_received_n(mock_server, 2).await;

    // Verify the second request hit the streaming path.
    let reqs = mock_server
        .received_requests()
        .await
        .expect("failed to fetch received requests");
    let stream_path = reqs[1].url.path();
    assert!(
        stream_path.ends_with("/converse-stream"),
        "streaming request must target /converse-stream, got: {}",
        stream_path
    );
    assert!(
        !stream_path.ends_with("/converse"),
        "streaming request must NOT target the non-streaming /converse endpoint"
    );
}

// ── Cross-region inference tests ──────────────────────────────────────────────

/// 8. When `cross_region_inference = true` and the provider region is
///    `us-east-1`, the proxy prepends `"us."` to the model ID in the upstream
///    request URL path.
///
/// The Bedrock `endpoint_url` function computes the effective model ID as
/// `"{prefix}.{model_id}"` where `prefix` is derived from the region
/// (`"us"` for `us-*`, `"eu"` for `eu-*`, `"ap"` for `ap-*`).
///
/// To exercise this path end-to-end we start the server manually with a custom
/// `ServerConfig` that has `cross_region_inference: true` and verify that the
/// wiremock server receives a request whose URL path contains the `"us."` prefix.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_bedrock_cross_region_inference_url_prefix() {
    // ── Start a wiremock server to act as the Bedrock upstream ────────────────
    let upstream = MockServer::start().await;

    // Match any POST to a path that looks like a Bedrock Converse call.
    // With cross-region enabled, the path will be
    // `/model/us.anthropic.claude-3-5-sonnet-20241022-v2:0/converse`.
    Mock::given(method("POST"))
        .and(path_regex(r"^/model/.+/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{ "text": "cross-region response" }]
                }
            },
            "stopReason": "end_turn",
            "usage": { "inputTokens": 8, "outputTokens": 4, "totalTokens": 12 }
        })))
        .mount(&upstream)
        .await;

    // ── Build a ServerConfig with cross_region_inference = true ───────────────
    let mut providers = HashMap::new();
    providers.insert(
        "bedrock".to_string(),
        ProviderConfig {
            // Point at the wiremock server instead of real AWS.
            base_url: Some(upstream.uri()),
            api_format: "bedrock".into(),
            models: vec!["anthropic.claude-3-5-sonnet-20241022-v2:0".into()],
            region: Some("us-east-1".into()),
            cross_region_inference: true, // the key flag under test
            project_id: None,
            timeout: "30s".into(),
            health_check_interval: "30s".into(),
            max_concurrent: 10,
            key_pool: KeyPoolConfig {
                selector: "weighted_random".into(),
                keys: vec![KeyEntry {
                    id: "bedrock-test-key".into(),
                    key_type: "static".into(),
                    api_key: Some(
                        r#"{"access_key":"AKIAIOSFODNN7EXAMPLE","secret_key":"test-secret","session_token":null,"region":"us-east-1"}"#.into(),
                    ),
                    role_arn: None,
                    region: None,
                    refresh_interval: None,
                    vault_path: None,
                    weight: 1.0,
                }],
            },
        },
    );

    // Single static-key auth validator — same pattern as build_test_config.
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

    let config = ServerConfig {
        providers,
        auth: AuthConfig { validators },
        admin: AdminConfig::default(),
        guardrails: GuardrailsConfig::default(),
        rate_limit: RateLimitConfig::default(),
        model_selection: ModelSelectionConfig::default(),
        routing: RoutingConfig::default(),
        ..ServerConfig::default()
    };

    // ── Start the server on a free port ───────────────────────────────────────
    let proxy_port = crate::port_allocator::allocate();
    let proxy_listener = tokio::net::TcpListener::bind(("127.0.0.1", proxy_port))
        .await
        .unwrap_or_else(|e| panic!("failed to bind proxy port {}: {}", proxy_port, e));
    let proxy_addr = proxy_listener.local_addr().unwrap();

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        switchboard_server::run_server(config, "", proxy_listener, None, async move {
            let _ = shutdown_rx.await;
        })
        .await
        .ok();
    });

    // Wait for the server to be ready.
    let http = reqwest::Client::new();
    let health_url = format!("http://{}/health", proxy_addr);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match http
            .get(&health_url)
            .header("Authorization", "Bearer test-api-key")
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => break,
            _ => {}
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("cross-region server did not become ready within 5 seconds");
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // ── Send a non-streaming chat request ────────────────────────────────────
    let body = json!({
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "cross-region test"}]
    });

    let resp = http
        .post(format!("http://{}/v1/chat/completions", proxy_addr))
        .header("Authorization", "Bearer test-api-key")
        .json(&body)
        .send()
        .await
        .expect("chat_completions request failed");

    assert_eq!(
        resp.status().as_u16(),
        200,
        "expected 200 OK from cross-region Bedrock proxy"
    );

    // ── Inspect the upstream request path ────────────────────────────────────
    let reqs = upstream
        .received_requests()
        .await
        .expect("failed to fetch upstream requests");

    assert_eq!(
        reqs.len(),
        1,
        "expected exactly one request to the Bedrock mock"
    );

    let path = reqs[0].url.path();

    // The path must contain the regional prefix "us." before the model ID,
    // producing `.../us.anthropic.claude-3-5-sonnet-20241022-v2:0/converse`.
    assert!(
        path.contains("/us.anthropic.claude-3-5-sonnet-20241022-v2:0/converse"),
        "cross-region path must contain 'us.' prefix on model ID, got: {path}"
    );

    // Confirm no bare (non-prefixed) path was used.
    assert!(
        !path.contains("/model/anthropic.claude-3-5-sonnet-20241022-v2:0/converse"),
        "cross-region path must NOT use bare (non-prefixed) model ID, got: {path}"
    );

    // Shut down the server.
    let _ = shutdown_tx.send(());
}

/// 9. When `cross_region_inference = false` (the default), the model ID in the
///    upstream URL path must NOT have any regional prefix.
///
/// This is the complement of test 8: both true and false cases are exercised
/// explicitly so that any future regression in the flag plumbing is caught.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_bedrock_no_cross_region_no_url_prefix() {
    // Standard harness — cross_region_inference defaults to false in
    // build_test_config.
    let harness = TestHarnessBuilder::new().with_bedrock().build().await;

    let mock_server = harness
        .mocks
        .bedrock
        .as_ref()
        .expect("bedrock mock must be present");
    crate::mocks::bedrock::mock_converse_ok("no-prefix response")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "no-prefix test"}]
    });

    harness.client.chat_completions(body).await;

    let reqs = mock_server
        .received_requests()
        .await
        .expect("failed to fetch received requests");

    assert_eq!(reqs.len(), 1, "expected exactly one upstream request");

    let path = reqs[0].url.path();

    // With cross_region_inference = false, the path must be the bare model ID
    // with no regional prefix.
    assert!(
        path.contains("/anthropic.claude-3-5-sonnet-20241022-v2:0/converse"),
        "non-cross-region path must NOT have a regional prefix, got: {path}"
    );
    assert!(
        !path.contains("/us.anthropic."),
        "non-cross-region path must NOT contain 'us.' prefix, got: {path}"
    );
    assert!(
        !path.contains("/eu.anthropic."),
        "non-cross-region path must NOT contain 'eu.' prefix, got: {path}"
    );
    assert!(
        !path.contains("/ap.anthropic."),
        "non-cross-region path must NOT contain 'ap.' prefix, got: {path}"
    );
}
