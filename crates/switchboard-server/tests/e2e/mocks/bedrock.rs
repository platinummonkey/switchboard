//! Wiremock mock templates for the AWS Bedrock Converse API.
//!
//! The Bedrock provider performs real SigV4 signing before sending requests,
//! but this mock does not validate the signature — it matches on HTTP method
//! and path regex only.
//!
//! Paths follow the pattern `/model/{model_id}/converse` (non-streaming)
//! and `/model/{model_id}/converse-stream` (streaming).

use serde_json::json;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, ResponseTemplate};

/// Non-streaming Converse API response.
pub fn mock_converse_ok(content: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path_regex(r"^/model/.+/converse$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{ "text": content }]
                }
            },
            "stopReason": "end_turn",
            "metrics": { "latencyMs": 42 },
            "usage": { "inputTokens": 10, "outputTokens": 5, "totalTokens": 15 }
        })))
}

/// Error response from Bedrock (e.g., ThrottlingException, ModelNotReadyException).
pub fn mock_converse_error(status: u16, error_type: &str, message: &str) -> Mock {
    Mock::given(method("POST"))
        .and(path_regex(r"^/model/.+/converse$"))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({
            "__type": error_type,
            "message": message
        })))
}
