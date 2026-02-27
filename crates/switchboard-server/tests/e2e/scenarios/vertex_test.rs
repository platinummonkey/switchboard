//! E2E scenario: Vertex AI (Gemini) provider.
//!
//! Tests cover:
//! - Basic chat via the Gemini generateContent endpoint
//! - Proxy converts Gemini response format to OpenAI format
//! - Streaming via streamGenerateContent returns an SSE response
//! - Upstream errors are surfaced to the caller

use serde_json::json;

use crate::assertions::{assert_openai_chat_response, assert_sse_content_type};
use crate::harness::TestHarnessBuilder;
use crate::mocks::vertex;

/// 1. A basic chat request routed to Vertex returns 200 and the correct content.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_vertex_generate_basic_chat() {
    let harness = TestHarnessBuilder::new().with_vertex().build().await;

    let mock_server = harness
        .mocks
        .vertex
        .as_ref()
        .expect("vertex mock must be present");
    vertex::mock_generate_ok("Rome").mount(mock_server).await;

    let body = json!({
        "model": "gemini-1.5-pro",
        "messages": [{"role": "user", "content": "Capital of Italy?"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "expected 200 OK from Vertex proxy"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["choices"][0]["message"]["content"], "Rome",
        "content should be 'Rome' as returned by the Vertex mock"
    );
}

/// 2. The proxy converts the Gemini generateContent response into the OpenAI
///    chat completion format, including `choices` and `usage` fields.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_vertex_response_converted_to_openai_format() {
    let harness = TestHarnessBuilder::new().with_vertex().build().await;

    let mock_server = harness
        .mocks
        .vertex
        .as_ref()
        .expect("vertex mock must be present");
    vertex::mock_generate_ok("Converted vertex response")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gemini-1.5-pro",
        "messages": [{"role": "user", "content": "Say something."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");

    assert_openai_chat_response(&json, "Converted vertex response");

    // `choices` must be present and non-empty.
    let choices = json["choices"]
        .as_array()
        .expect("choices must be an array");
    assert!(!choices.is_empty(), "choices must not be empty");

    // `usage` must be present.
    let usage = &json["usage"];
    assert!(
        !usage.is_null(),
        "usage must be present in the OpenAI-format response"
    );
}

/// 3. A streaming request to the Vertex provider returns a `text/event-stream`
///    SSE response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_vertex_streaming() {
    let harness = TestHarnessBuilder::new().with_vertex().build().await;

    let mock_server = harness
        .mocks
        .vertex
        .as_ref()
        .expect("vertex mock must be present");
    vertex::mock_stream_generate_ok("hello")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gemini-1.5-pro",
        "messages": [{"role": "user", "content": "Stream something."}]
    });

    let resp = harness.client.chat_completions_stream(body).await;

    // The proxy must return a streaming SSE response.
    assert_sse_content_type(&resp);
}

/// 4. When the Vertex upstream responds with an error (e.g., RESOURCE_EXHAUSTED),
///    the proxy returns a non-200 status to the caller.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_vertex_upstream_error() {
    let harness = TestHarnessBuilder::new().with_vertex().build().await;

    let mock_server = harness
        .mocks
        .vertex
        .as_ref()
        .expect("vertex mock must be present");
    vertex::mock_generate_error(429, "RESOURCE_EXHAUSTED")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gemini-1.5-pro",
        "messages": [{"role": "user", "content": "Capital of Italy?"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_ne!(
        resp.status().as_u16(),
        200,
        "a Vertex upstream error must not be returned as 200"
    );
}
