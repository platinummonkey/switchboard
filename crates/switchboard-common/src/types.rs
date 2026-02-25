use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Message primitives ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    /// Tool call id — populated for role=tool responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls — populated for assistant messages that invoke tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Return the concatenated text, useful for token counting and classification.
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| {
                    if let ContentPart::Text { text } = p {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { source: ImageSource },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

// ── Tools ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
}

// ── Usage / cost tracking ────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
}

impl Usage {
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

// ── Request context ───────────────────────────────────────────────────────────

/// Metadata extracted from an incoming proxy request.
/// Passed to key selectors, routing, and observability.
#[derive(Debug, Clone, Default)]
pub struct RequestContext {
    /// Unique ID for this request (generated at ingress).
    pub request_id: String,
    /// Resolved user identifier, if available.
    pub user_id: Option<String>,
    /// Resolved team, if available.
    pub team: Option<String>,
    /// Requested or resolved model name.
    pub model: Option<String>,
    /// Client tool name (e.g. "claude-code", "cursor").
    pub tool: Option<String>,
    /// Session identifier for sticky key selection.
    pub session_id: Option<String>,
    /// Raw Switchboard protocol headers from the request.
    pub switchboard_headers: HashMap<String, String>,
}

impl RequestContext {
    pub fn new() -> Self {
        Self {
            request_id: Uuid::new_v4().to_string(),
            ..Default::default()
        }
    }
}

// ── Proxied request/response ──────────────────────────────────────────────────

/// Normalized internal representation of a chat/completion request.
/// Provider modules convert this to/from their wire formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxiedRequest {
    /// The model to use (may be rewritten by routing before forwarding).
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Whether the client requested streaming.
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Provider-specific extra fields passed through verbatim.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Normalized internal representation of a completed (non-streaming) response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxiedResponse {
    pub model: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_content_as_text_string() {
        let m = Message {
            role: Role::User,
            content: MessageContent::Text("hello".into()),
            tool_call_id: None,
            tool_calls: None,
        };
        assert_eq!(m.content.as_text(), "hello");
    }

    #[test]
    fn test_message_content_as_text_parts() {
        let parts = vec![
            ContentPart::Text {
                text: "part one".into(),
            },
            ContentPart::Text {
                text: "part two".into(),
            },
        ];
        let content = MessageContent::Parts(parts);
        assert_eq!(content.as_text(), "part one\npart two");
    }

    #[test]
    fn test_usage_total_tokens() {
        let u = Usage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        assert_eq!(u.total_tokens(), 150);
    }

    #[test]
    fn test_request_context_new_has_request_id() {
        let ctx = RequestContext::new();
        assert!(!ctx.request_id.is_empty());
    }

    #[test]
    fn test_request_context_unique_ids() {
        let a = RequestContext::new();
        let b = RequestContext::new();
        assert_ne!(a.request_id, b.request_id);
    }

    #[test]
    fn test_proxied_request_roundtrip() {
        let req = ProxiedRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("hello".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: None,
            system: None,
            extra: HashMap::new(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: ProxiedRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.model, req.model);
        assert_eq!(decoded.max_tokens, req.max_tokens);
    }

    #[test]
    fn test_role_serde_lowercase() {
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
    }
}
