//! Wiremock mock templates for the Anthropic Messages API.

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

/// Standard non-streaming messages response.
pub fn mock_messages_ok(content: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "text", "text": content }],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        })))
}

/// Streaming messages response with a single text delta.
pub fn mock_messages_streaming(text_delta: &str) -> Mock {
    // Build a minimal valid Anthropic SSE stream.
    let body = format!(
        "event: message_start\n\
         data: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_test\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-5-sonnet-20241022\",\"stop_reason\":null,\"usage\":{{\"input_tokens\":10,\"output_tokens\":0}}}}}}\n\n\
         event: content_block_start\n\
         data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
         event: content_block_delta\n\
         data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{text_delta}\"}}}}\n\n\
         event: content_block_stop\n\
         data: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
         event: message_delta\n\
         data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}},\"usage\":{{\"output_tokens\":5}}}}\n\n\
         event: message_stop\n\
         data: {{\"type\":\"message_stop\"}}\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .insert_header("cache-control", "no-cache")
                .set_body_string(body),
        )
}

/// Response containing a `tool_use` content block (Anthropic function calling).
///
/// `tool_use_id` is the ID the client must echo back in the `tool_result`.
/// `input` is the JSON object the model wants to pass to the tool.
pub fn mock_messages_with_tool_use(
    tool_name: &str,
    tool_use_id: &str,
    input: serde_json::Value,
) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_tool_test",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_name,
                "input": input
            }],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 20, "output_tokens": 15 }
        })))
}

/// Error response from Anthropic.
pub fn mock_messages_error(status: u16, error_type: &str, message: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({
            "type": "error",
            "error": { "type": error_type, "message": message }
        })))
}
