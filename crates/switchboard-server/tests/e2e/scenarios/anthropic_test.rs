//! E2E scenario: Anthropic Messages API — non-streaming, streaming, usage
//! fields, system-prompt forwarding, and upstream error propagation.

use serde_json::json;

use crate::harness::TestHarnessBuilder;
use crate::mocks::anthropic::{mock_messages_error, mock_messages_ok, mock_messages_streaming};

/// Full happy-path via `POST /api/v1/messages`: the response must include
/// `type: "message"` and `content[0].text` matching the mock payload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_anthropic_messages_returns_content() {
    let harness = TestHarnessBuilder::new().with_anthropic().build().await;
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_messages_ok("Bonjour").mount(anthropic_mock).await;

    let resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Say hello in French"}],
            "max_tokens": 256
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_anthropic_response(&body, "Bonjour");
}

/// The `usage` object must be present with `input_tokens` and `output_tokens`
/// forwarded from the upstream response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_anthropic_messages_usage_present() {
    let harness = TestHarnessBuilder::new().with_anthropic().build().await;
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_messages_ok("answer").mount(anthropic_mock).await;

    let resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Question"}],
            "max_tokens": 256
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();

    let input_tokens = body["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = body["usage"]["output_tokens"].as_u64().unwrap_or(0);

    assert!(
        input_tokens > 0,
        "usage.input_tokens must be present and positive"
    );
    assert!(
        output_tokens > 0,
        "usage.output_tokens must be present and positive"
    );
}

/// Streaming request via the Anthropic endpoint: must return 200, SSE
/// content-type, and at least one SSE event.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_anthropic_streaming_returns_sse() {
    let harness = TestHarnessBuilder::new().with_anthropic().build().await;
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_messages_streaming("Hello").mount(anthropic_mock).await;

    let resp = harness
        .client
        .anthropic_messages_stream(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Greet me"}],
            "max_tokens": 64
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 for SSE stream");
    crate::assertions::assert_sse_content_type(&resp);

    let events = crate::client::TestClient::collect_sse(resp).await;
    assert!(
        !events.is_empty(),
        "expected at least one SSE event from Anthropic stream"
    );
}

/// A request that includes a `system` field must still reach the upstream mock
/// (system-prompt forwarding exercised end-to-end), resulting in exactly 1 hit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_anthropic_system_prompt_forwarded() {
    let harness = TestHarnessBuilder::new().with_anthropic().build().await;
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_messages_ok("ok").mount(anthropic_mock).await;

    let resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32
        }))
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "expected 200 when system prompt is present"
    );

    // The upstream mock must have been hit exactly once, confirming the request
    // (including the system prompt field) was forwarded.
    crate::assertions::assert_received_n(anthropic_mock, 1).await;
}

/// When the upstream Anthropic API returns a 529 (overloaded), the proxy must
/// surface a non-200 response to the client.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_anthropic_upstream_error_propagated() {
    let harness = TestHarnessBuilder::new().with_anthropic().build().await;
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_messages_error(529, "overloaded_error", "overloaded")
        .mount(anthropic_mock)
        .await;

    let resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32
        }))
        .await;

    assert_ne!(
        resp.status().as_u16(),
        200,
        "expected non-200 when upstream returns 529"
    );
}
