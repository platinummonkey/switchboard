//! Common assertion helpers for E2E tests.

use reqwest::Response;
use serde_json::Value;
use wiremock::MockServer;

/// Assert that a wiremock server received exactly `n` requests.
pub async fn assert_received_n(mock_server: &MockServer, n: usize) {
    let reqs = mock_server
        .received_requests()
        .await
        .expect("failed to fetch received requests");
    assert_eq!(
        reqs.len(),
        n,
        "expected {} request(s) on mock server, got {}",
        n,
        reqs.len()
    );
}

/// Assert that the response has `Content-Type: text/event-stream`.
pub fn assert_sse_content_type(resp: &Response) {
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("text/event-stream"),
        "expected SSE content-type, got: {:?}",
        ct
    );
}

/// Assert that a JSON body looks like a valid OpenAI chat completion.
pub fn assert_openai_chat_response(body: &Value, expected_content: &str) {
    assert_eq!(
        body["choices"][0]["message"]["content"], expected_content,
        "unexpected content in OpenAI response"
    );
    let total = body["usage"]["total_tokens"].as_u64().unwrap_or(0);
    assert!(total > 0, "expected non-zero total_tokens in usage");
}

/// Assert that a JSON body looks like a valid Anthropic messages response.
pub fn assert_anthropic_response(body: &Value, expected_text: &str) {
    assert_eq!(body["type"], "message", "expected type=message");
    assert_eq!(body["role"], "assistant", "expected role=assistant");
    assert_eq!(
        body["content"][0]["text"], expected_text,
        "unexpected text in Anthropic response"
    );
}
