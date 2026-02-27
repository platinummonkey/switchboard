//! E2E scenario: OpenAI-compatible endpoint — non-streaming, streaming, error
//! propagation, and usage field verification.

use serde_json::json;

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai::{mock_chat_error, mock_chat_ok, mock_chat_streaming};

/// Full happy-path: mount mock, send non-streaming request, parse JSON, verify
/// `choices[0].message.content` matches the mock response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_openai_non_streaming_returns_content() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "The Eiffel Tower")
        .mount(openai_mock)
        .await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Describe Paris"}]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_openai_chat_response(&body, "The Eiffel Tower");
}

/// The `usage` object returned by OpenAI must be forwarded intact with all
/// three token count fields present and positive.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_openai_preserves_usage_in_response() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "answer").mount(openai_mock).await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Question"}]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();

    let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    let total_tokens = body["usage"]["total_tokens"].as_u64().unwrap_or(0);

    assert!(
        prompt_tokens > 0,
        "usage.prompt_tokens must be present and positive"
    );
    assert!(
        completion_tokens > 0,
        "usage.completion_tokens must be present and positive"
    );
    assert!(
        total_tokens > 0,
        "usage.total_tokens must be present and positive"
    );
}

/// Streaming request: proxy must return 200 with `Content-Type: text/event-stream`
/// and the collected SSE payloads must equal the 2 chunks sent by the mock.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_openai_streaming_returns_sse() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    let chunk1 = r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}"#;
    let chunk2 = r#"{"id":"c2","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"index":0}]}"#;

    mock_chat_streaming(&[chunk1, chunk2])
        .mount(openai_mock)
        .await;

    let resp = harness
        .client
        .chat_completions_stream(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Say hello"}]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 for streaming");
    crate::assertions::assert_sse_content_type(&resp);

    let events = crate::client::TestClient::collect_sse(resp).await;
    assert_eq!(
        events.len(),
        2,
        "expected exactly 2 SSE data events, got {}",
        events.len()
    );
}

/// The chunks must arrive in order: first contains "Hello", second " world".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_openai_streaming_chunks_in_order() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    let chunk1 = r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}"#;
    let chunk2 = r#"{"id":"c2","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"index":0}]}"#;

    mock_chat_streaming(&[chunk1, chunk2])
        .mount(openai_mock)
        .await;

    let resp = harness
        .client
        .chat_completions_stream(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Say hello world"}]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200);

    let events = crate::client::TestClient::collect_sse(resp).await;
    assert!(
        events.len() >= 2,
        "expected at least 2 SSE events, got {}",
        events.len()
    );

    // Parse each chunk and extract the delta content.
    let first: serde_json::Value =
        serde_json::from_str(&events[0]).expect("first SSE chunk is not valid JSON");
    let second: serde_json::Value =
        serde_json::from_str(&events[1]).expect("second SSE chunk is not valid JSON");

    let first_content = first["choices"][0]["delta"]["content"]
        .as_str()
        .unwrap_or("");
    let second_content = second["choices"][0]["delta"]["content"]
        .as_str()
        .unwrap_or("");

    assert!(
        first_content.contains("Hello"),
        "first chunk should contain 'Hello', got {:?}",
        first_content
    );
    assert!(
        second_content.contains("world"),
        "second chunk should contain 'world', got {:?}",
        second_content
    );
}

/// When the upstream returns a 429, the proxy must propagate a non-200 status
/// to the client rather than masking the error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_openai_upstream_error_propagated() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_error(429, "rate limit exceeded")
        .mount(openai_mock)
        .await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .await;

    assert_ne!(
        resp.status().as_u16(),
        200,
        "expected non-200 when upstream returns 429"
    );
}
