//! Wiremock mock templates for the Vertex AI (Gemini) API.
//!
//! The Vertex provider sends an OAuth bearer token as `Authorization: Bearer
//! <token>`. This mock does NOT validate the token — it matches on HTTP
//! method and path suffix only. Tests configure a static token as the pool
//! key; the mock simply ignores it.
//!
//! Paths follow the pattern:
//! - Non-streaming: `.../{model}:generateContent`
//! - Streaming:     `.../{model}:streamGenerateContent`

use serde_json::json;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, ResponseTemplate};

/// Non-streaming generateContent response (Gemini format).
pub fn mock_generate_ok(text: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path_regex(r":generateContent$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {
                    "parts": [{ "text": text }],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": []
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        })))
}

/// Streaming streamGenerateContent SSE response.
pub fn mock_stream_generate_ok(text: &str) -> Mock {
    let chunk = json!({
        "candidates": [{
            "content": {
                "parts": [{ "text": text }],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    });
    let body = format!("data: {}\n\n", chunk);
    Mock::given(method("POST"))
        .and(path_regex(r":streamGenerateContent"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
}

/// Error response from Vertex (e.g., quota exceeded, invalid model).
pub fn mock_generate_error(status: u16, message: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path_regex(r":generateContent$"))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({
            "error": {
                "code": status,
                "message": message,
                "status": "RESOURCE_EXHAUSTED"
            }
        })))
}
