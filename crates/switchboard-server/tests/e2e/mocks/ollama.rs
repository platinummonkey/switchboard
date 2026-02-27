//! Wiremock mock templates for the Ollama OpenAI-compatible API.
//!
//! Ollama in openai-compat mode exposes `/v1/chat/completions`, so these
//! mocks are identical in shape to the OpenAI mocks but with different
//! model names and usage values that reflect Ollama's output.

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, ResponseTemplate};

/// Non-streaming chat completion (Ollama openai-compat format).
pub fn mock_chat_ok(model: &str, content: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-ollama-test",
            "object": "chat.completion",
            "created": 1_700_000_000_u64,
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12
            }
        })))
}

/// Streaming chat response (Ollama openai-compat format).
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
                .set_body_string(body),
        )
}

/// Error response from Ollama (model not found).
#[allow(dead_code)]
pub fn mock_model_not_found(model: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": format!("model '{}' not found", model)
        })))
}
