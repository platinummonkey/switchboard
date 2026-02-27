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

/// Response containing one or more tool calls (function calling).
///
/// `calls` is a slice of `(call_id, function_name, arguments_json_string)`.
/// The response uses `finish_reason: "tool_calls"` with a null content body.
pub fn mock_chat_with_tool_calls(calls: &[(&str, &str, &str)]) -> Mock {
    let tool_calls: Vec<serde_json::Value> = calls
        .iter()
        .map(|(id, name, args)| {
            json!({
                "id": id,
                "type": "function",
                "function": { "name": name, "arguments": args }
            })
        })
        .collect();

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-tool-test",
            "object": "chat.completion",
            "created": 1_700_000_000_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": tool_calls
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35 }
        })))
}

/// Tool result follow-up response after a tool call has been executed.
/// Returns a normal assistant message as the final response.
pub fn mock_chat_after_tool(content: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-tool-result",
            "object": "chat.completion",
            "created": 1_700_000_000_u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40 }
        })))
}

/// Models list response for health checks.
#[allow(dead_code)]
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
