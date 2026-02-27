//! E2E scenario: Bedrock Converse API.
//!
//! Tests cover:
//! - Basic chat via the Bedrock Converse endpoint
//! - Proxy converts Bedrock response format to OpenAI format
//! - Upstream errors are surfaced to the caller
//! - Mock server receives exactly the expected number of requests

use serde_json::json;

use crate::assertions::{assert_openai_chat_response, assert_received_n};
use crate::harness::TestHarnessBuilder;
use crate::mocks::bedrock;

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
