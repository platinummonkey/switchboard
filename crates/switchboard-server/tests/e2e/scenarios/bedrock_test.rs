//! E2E scenario: Bedrock Converse API.
//!
//! Tests cover:
//! - Basic chat via the Bedrock Converse endpoint
//! - Proxy converts Bedrock response format to OpenAI format
//! - Upstream errors are surfaced to the caller
//! - Mock server receives exactly the expected number of requests
//! - Streaming requests hit the `/converse-stream` path (not `/converse`)
//! - Non-streaming and streaming requests use distinct upstream endpoints

use serde_json::json;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, ResponseTemplate};

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
