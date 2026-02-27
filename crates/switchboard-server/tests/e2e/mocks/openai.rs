//! Wiremock mock templates for the OpenAI API.

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

/// Standard non-streaming chat completion response.
pub fn mock_chat_ok(model: &str, content: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000_u64,
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        })))
}

/// Streaming chat response with provided SSE data payloads.
///
/// Each string in `chunks` becomes a `data: <chunk>\n\n` line.
/// A final `data: [DONE]\n\n` is appended automatically.
pub fn mock_chat_streaming(chunks: &[&str]) -> Mock {
    let body: String = chunks
        .iter()
        .map(|c| format!("data: {}\n\n", c))
        .collect::<String>()
        + "data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .insert_header("cache-control", "no-cache")
                .set_body_string(body),
        )
}

/// Error response (e.g., 400 invalid request, 429 rate limit, 500 server error).
pub fn mock_chat_error(status: u16, message: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": null
            }
        })))
}

/// Models list response for health checks.
pub fn mock_models_ok() -> Mock {
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                { "id": "gpt-4o", "object": "model", "created": 1_700_000_000_u64 },
                { "id": "gpt-3.5-turbo", "object": "model", "created": 1_700_000_000_u64 }
            ]
        })))
}
