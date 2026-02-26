//! Format transformations between OpenAI/Anthropic wire formats and the
//! internal `ProxiedRequest`/`ProxiedResponse` types.

use switchboard_common::types::{
    FunctionCall, Message, MessageContent, ProxiedRequest, ProxiedResponse, Role, Tool, ToolCall,
};
use uuid::Uuid;

use crate::proxy::error::ProxyError;

// ── OpenAI → internal ────────────────────────────────────────────────────────

/// Parse an incoming OpenAI chat completion JSON body into [`ProxiedRequest`].
///
/// # Errors
/// Returns [`ProxyError::InvalidRequest`] if required fields are missing or
/// have wrong types.
pub fn openai_to_proxied(body: &serde_json::Value) -> Result<ProxiedRequest, ProxyError> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::InvalidRequest("missing required field: model".into()))?
        .to_string();

    let messages_raw = body
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProxyError::InvalidRequest("missing required field: messages".into()))?;

    let mut messages = Vec::with_capacity(messages_raw.len());
    for (i, msg) in messages_raw.iter().enumerate() {
        messages.push(parse_openai_message(msg, i)?);
    }

    let tools = parse_openai_tools(body)?;
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let temperature = body.get("temperature").and_then(|v| v.as_f64());
    let top_p = body.get("top_p").and_then(|v| v.as_f64());

    // Extract system message if present as a separate system message in the
    // messages array; keep it in `system` field for cross-provider compat.
    let system = extract_system_from_openai(body, &messages);

    Ok(ProxiedRequest {
        model,
        messages,
        tools,
        stream,
        max_tokens,
        temperature,
        top_p,
        system,
        extra: Default::default(),
    })
}

/// Parse a single OpenAI message object.
fn parse_openai_message(msg: &serde_json::Value, index: usize) -> Result<Message, ProxyError> {
    let role_str = msg.get("role").and_then(|v| v.as_str()).ok_or_else(|| {
        ProxyError::InvalidRequest(format!("messages[{index}]: missing role field"))
    })?;

    let role = match role_str {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => {
            return Err(ProxyError::InvalidRequest(format!(
                "messages[{index}]: unknown role '{other}'"
            )));
        }
    };

    let content = parse_openai_content(msg, index)?;

    let tool_call_id = msg
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let tool_calls = parse_openai_tool_calls(msg)?;

    Ok(Message {
        role,
        content,
        tool_call_id,
        tool_calls,
    })
}

/// Parse the content field which may be a string or an array of parts.
fn parse_openai_content(
    msg: &serde_json::Value,
    index: usize,
) -> Result<MessageContent, ProxyError> {
    match msg.get("content") {
        Some(serde_json::Value::String(s)) => Ok(MessageContent::Text(s.clone())),
        Some(serde_json::Value::Array(_)) => {
            // For simplicity, coerce parts back to text.
            let text = msg["content"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|p| {
                    if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                        p.get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok(MessageContent::Text(text))
        }
        Some(serde_json::Value::Null) | None => {
            // Content can be null for assistant messages with tool calls.
            Ok(MessageContent::Text(String::new()))
        }
        Some(other) => Err(ProxyError::InvalidRequest(format!(
            "messages[{index}]: unexpected content type: {}",
            other
        ))),
    }
}

/// Parse optional tool_calls array from an assistant message.
fn parse_openai_tool_calls(msg: &serde_json::Value) -> Result<Option<Vec<ToolCall>>, ProxyError> {
    let Some(arr) = msg.get("tool_calls").and_then(|v| v.as_array()) else {
        return Ok(None);
    };

    let mut calls = Vec::with_capacity(arr.len());
    for tc in arr {
        let id = tc
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let call_type = tc
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("function")
            .to_string();
        let func = tc.get("function").unwrap_or(&serde_json::Value::Null);
        let name = func
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let arguments = func
            .get("arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("{}")
            .to_string();
        calls.push(ToolCall {
            id,
            call_type,
            function: FunctionCall { name, arguments },
        });
    }
    Ok(Some(calls))
}

/// Parse the top-level `tools` array if present.
fn parse_openai_tools(body: &serde_json::Value) -> Result<Option<Vec<Tool>>, ProxyError> {
    let Some(arr) = body.get("tools").and_then(|v| v.as_array()) else {
        return Ok(None);
    };

    let mut tools = Vec::with_capacity(arr.len());
    for t in arr {
        // OpenAI tool format: {"type": "function", "function": {...}}
        let func = match t.get("function") {
            Some(f) => f,
            None => t, // Some clients send the function fields directly.
        };
        let name = func
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ProxyError::InvalidRequest("tool missing name field".into()))?
            .to_string();
        let description = func
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let input_schema = func
            .get("parameters")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        tools.push(Tool {
            name,
            description,
            input_schema,
        });
    }
    Ok(Some(tools))
}

/// Extract the system instruction from the body's `system` field or from a
/// leading `system` role message. Returns `None` if no system content found.
fn extract_system_from_openai(body: &serde_json::Value, _messages: &[Message]) -> Option<String> {
    // Some OpenAI clients send a top-level `system` field (non-standard but
    // common).  Prefer that.
    if let Some(s) = body.get("system").and_then(|v| v.as_str()) {
        return Some(s.to_string());
    }
    None
}

// ── internal → OpenAI ────────────────────────────────────────────────────────

/// Serialize [`ProxiedResponse`] back to OpenAI chat completion JSON format.
pub fn proxied_to_openai(resp: &ProxiedResponse) -> serde_json::Value {
    let usage = resp.usage.as_ref().map(|u| {
        serde_json::json!({
            "prompt_tokens": u.input_tokens,
            "completion_tokens": u.output_tokens,
            "total_tokens": u.total_tokens(),
        })
    });

    let message = serde_json::json!({
        "role": "assistant",
        "content": resp.content,
    });

    serde_json::json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4().simple()),
        "object": "chat.completion",
        "model": resp.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": resp.finish_reason.as_deref().unwrap_or("stop"),
        }],
        "usage": usage,
    })
}

// ── Anthropic → internal ─────────────────────────────────────────────────────

/// Parse an incoming Anthropic Messages API JSON body into [`ProxiedRequest`].
///
/// # Errors
/// Returns [`ProxyError::InvalidRequest`] if required fields are missing or
/// have wrong types.
pub fn anthropic_to_proxied(body: &serde_json::Value) -> Result<ProxiedRequest, ProxyError> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::InvalidRequest("missing required field: model".into()))?
        .to_string();

    let messages_raw = body
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProxyError::InvalidRequest("missing required field: messages".into()))?;

    let mut messages = Vec::with_capacity(messages_raw.len());
    for (i, msg) in messages_raw.iter().enumerate() {
        messages.push(parse_anthropic_message(msg, i)?);
    }

    let tools = parse_anthropic_tools(body)?;
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let temperature = body.get("temperature").and_then(|v| v.as_f64());
    let top_p = body.get("top_p").and_then(|v| v.as_f64());
    let system = body
        .get("system")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok(ProxiedRequest {
        model,
        messages,
        tools,
        stream,
        max_tokens,
        temperature,
        top_p,
        system,
        extra: Default::default(),
    })
}

/// Parse a single Anthropic message object.
fn parse_anthropic_message(msg: &serde_json::Value, index: usize) -> Result<Message, ProxyError> {
    let role_str = msg.get("role").and_then(|v| v.as_str()).ok_or_else(|| {
        ProxyError::InvalidRequest(format!("messages[{index}]: missing role field"))
    })?;

    let role = match role_str {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        other => {
            return Err(ProxyError::InvalidRequest(format!(
                "messages[{index}]: unknown Anthropic role '{other}'"
            )));
        }
    };

    let content = parse_anthropic_content(msg, index)?;

    Ok(Message {
        role,
        content,
        tool_call_id: None,
        tool_calls: None,
    })
}

/// Parse Anthropic content which may be a string or an array of content blocks.
fn parse_anthropic_content(
    msg: &serde_json::Value,
    _index: usize,
) -> Result<MessageContent, ProxyError> {
    match msg.get("content") {
        Some(serde_json::Value::String(s)) => Ok(MessageContent::Text(s.clone())),
        Some(serde_json::Value::Array(parts)) => {
            // Extract text from text-type content blocks.
            let text = parts
                .iter()
                .filter_map(|p| {
                    if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                        p.get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok(MessageContent::Text(text))
        }
        Some(serde_json::Value::Null) | None => Ok(MessageContent::Text(String::new())),
        Some(other) => Err(ProxyError::InvalidRequest(format!(
            "unexpected content type: {}",
            other
        ))),
    }
}

/// Parse Anthropic native tool definitions.
fn parse_anthropic_tools(body: &serde_json::Value) -> Result<Option<Vec<Tool>>, ProxyError> {
    let Some(arr) = body.get("tools").and_then(|v| v.as_array()) else {
        return Ok(None);
    };

    let mut tools = Vec::with_capacity(arr.len());
    for t in arr {
        let name = t
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ProxyError::InvalidRequest("tool missing name field".into()))?
            .to_string();
        let description = t
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let input_schema = t
            .get("input_schema")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        tools.push(Tool {
            name,
            description,
            input_schema,
        });
    }
    Ok(Some(tools))
}

// ── internal → Anthropic ─────────────────────────────────────────────────────

/// Serialize [`ProxiedResponse`] back to Anthropic Messages API JSON format.
pub fn proxied_to_anthropic(resp: &ProxiedResponse) -> serde_json::Value {
    let usage = resp.usage.as_ref().map(|u| {
        serde_json::json!({
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
        })
    });

    serde_json::json!({
        "id": format!("msg_{}", Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": resp.content}],
        "model": resp.model,
        "stop_reason": resp.finish_reason.as_deref().unwrap_or("end_turn"),
        "usage": usage,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use switchboard_common::types::Usage;

    // ── openai_to_proxied ────────────────────────────────────────────────────

    #[test]
    fn test_openai_to_proxied_basic() {
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello, world!"}
            ]
        });
        let req = openai_to_proxied(&body).unwrap();
        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].content.as_text(), "Hello, world!");
        assert!(!req.stream);
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn test_openai_to_proxied_with_all_fields() {
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"}
            ],
            "stream": true,
            "max_tokens": 512,
            "temperature": 0.5,
            "top_p": 0.9
        });
        let req = openai_to_proxied(&body).unwrap();
        assert!(req.stream);
        assert_eq!(req.max_tokens, Some(512));
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.messages[0].role, Role::System);
    }

    #[test]
    fn test_openai_to_proxied_missing_model_errors() {
        let body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        let err = openai_to_proxied(&body).unwrap_err();
        assert!(matches!(err, ProxyError::InvalidRequest(_)));
        assert!(err.to_string().contains("model"));
    }

    #[test]
    fn test_openai_to_proxied_missing_messages_errors() {
        let body = serde_json::json!({"model": "gpt-4o"});
        let err = openai_to_proxied(&body).unwrap_err();
        assert!(matches!(err, ProxyError::InvalidRequest(_)));
    }

    #[test]
    fn test_openai_to_proxied_unknown_role_errors() {
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "pirate", "content": "arr"}]
        });
        let err = openai_to_proxied(&body).unwrap_err();
        assert!(matches!(err, ProxyError::InvalidRequest(_)));
        assert!(err.to_string().contains("pirate"));
    }

    #[test]
    fn test_openai_to_proxied_with_tools() {
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Call a tool"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        });
        let req = openai_to_proxied(&body).unwrap();
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].description.as_deref(), Some("Get weather"));
    }

    #[test]
    fn test_openai_to_proxied_null_content_ok() {
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "assistant", "content": null, "tool_calls": []}]
        });
        let req = openai_to_proxied(&body).unwrap();
        assert_eq!(req.messages[0].content.as_text(), "");
    }

    // ── proxied_to_openai ────────────────────────────────────────────────────

    #[test]
    fn test_proxied_to_openai_basic() {
        let resp = ProxiedResponse {
            model: "gpt-4o".into(),
            content: "Hello!".into(),
            tool_calls: None,
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            }),
        };
        let json = proxied_to_openai(&resp);
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["prompt_tokens"], 10);
        assert_eq!(json["usage"]["completion_tokens"], 5);
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_proxied_to_openai_no_usage() {
        let resp = ProxiedResponse {
            model: "gpt-4o".into(),
            content: "Hi".into(),
            tool_calls: None,
            finish_reason: None,
            usage: None,
        };
        let json = proxied_to_openai(&resp);
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert!(json["usage"].is_null());
    }

    // ── anthropic_to_proxied ─────────────────────────────────────────────────

    #[test]
    fn test_anthropic_to_proxied_basic() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 1024
        });
        let req = anthropic_to_proxied(&body).unwrap();
        assert_eq!(req.model, "claude-sonnet-4-20250514");
        assert_eq!(req.max_tokens, Some(1024));
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].content.as_text(), "Hello!");
    }

    #[test]
    fn test_anthropic_to_proxied_with_system() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "You are a helpful assistant.",
            "max_tokens": 512
        });
        let req = anthropic_to_proxied(&body).unwrap();
        assert_eq!(req.system.as_deref(), Some("You are a helpful assistant."));
    }

    #[test]
    fn test_anthropic_to_proxied_content_blocks() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello from blocks"}]
            }],
            "max_tokens": 256
        });
        let req = anthropic_to_proxied(&body).unwrap();
        assert_eq!(req.messages[0].content.as_text(), "Hello from blocks");
    }

    #[test]
    fn test_anthropic_to_proxied_missing_model_errors() {
        let body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(anthropic_to_proxied(&body).is_err());
    }

    #[test]
    fn test_anthropic_to_proxied_unknown_role_errors() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "system", "content": "be helpful"}],
            "max_tokens": 100
        });
        let err = anthropic_to_proxied(&body).unwrap_err();
        assert!(matches!(err, ProxyError::InvalidRequest(_)));
    }

    // ── proxied_to_anthropic ─────────────────────────────────────────────────

    #[test]
    fn test_proxied_to_anthropic_basic() {
        let resp = ProxiedResponse {
            model: "claude-sonnet-4-20250514".into(),
            content: "I can help!".into(),
            tool_calls: None,
            finish_reason: Some("end_turn".into()),
            usage: Some(Usage {
                input_tokens: 20,
                output_tokens: 10,
                ..Default::default()
            }),
        };
        let json = proxied_to_anthropic(&resp);
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "I can help!");
        assert_eq!(json["stop_reason"], "end_turn");
        assert_eq!(json["usage"]["input_tokens"], 20);
        assert_eq!(json["usage"]["output_tokens"], 10);
    }

    #[test]
    fn test_proxied_to_anthropic_no_finish_reason() {
        let resp = ProxiedResponse {
            model: "claude-sonnet-4-20250514".into(),
            content: "response".into(),
            tool_calls: None,
            finish_reason: None,
            usage: None,
        };
        let json = proxied_to_anthropic(&resp);
        assert_eq!(json["stop_reason"], "end_turn");
    }

    // ── Insta snapshot tests ─────────────────────────────────────────────────

    #[test]
    fn test_snapshot_openai_to_proxied() {
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 256,
            "temperature": 0.7
        });
        let req = openai_to_proxied(&body).unwrap();
        // Serialize to JSON for snapshot comparison (skip extra field).
        let snapshot = serde_json::json!({
            "model": req.model,
            "messages": req.messages.iter().map(|m| serde_json::json!({
                "role": serde_json::to_value(&m.role).unwrap(),
                "content": m.content.as_text(),
            })).collect::<Vec<_>>(),
            "stream": req.stream,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        });
        insta::assert_json_snapshot!(snapshot);
    }

    #[test]
    fn test_snapshot_proxied_to_openai() {
        let resp = ProxiedResponse {
            model: "gpt-4o".into(),
            content: "4".into(),
            tool_calls: None,
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                input_tokens: 15,
                output_tokens: 3,
                ..Default::default()
            }),
        };
        let json = proxied_to_openai(&resp);
        // Snapshot everything except the random id.
        let snapshot = serde_json::json!({
            "object": json["object"],
            "model": json["model"],
            "choices": json["choices"],
            "usage": json["usage"],
        });
        insta::assert_json_snapshot!(snapshot);
    }

    #[test]
    fn test_snapshot_anthropic_to_proxied() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "system": "Be concise.",
            "max_tokens": 128
        });
        let req = anthropic_to_proxied(&body).unwrap();
        let snapshot = serde_json::json!({
            "model": req.model,
            "messages": req.messages.iter().map(|m| serde_json::json!({
                "role": serde_json::to_value(&m.role).unwrap(),
                "content": m.content.as_text(),
            })).collect::<Vec<_>>(),
            "system": req.system,
            "max_tokens": req.max_tokens,
        });
        insta::assert_json_snapshot!(snapshot);
    }

    #[test]
    fn test_snapshot_proxied_to_anthropic() {
        let resp = ProxiedResponse {
            model: "claude-sonnet-4-20250514".into(),
            content: "4".into(),
            tool_calls: None,
            finish_reason: Some("end_turn".into()),
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 2,
                ..Default::default()
            }),
        };
        let json = proxied_to_anthropic(&resp);
        // Snapshot everything except the random id.
        let snapshot = serde_json::json!({
            "type": json["type"],
            "role": json["role"],
            "content": json["content"],
            "model": json["model"],
            "stop_reason": json["stop_reason"],
            "usage": json["usage"],
        });
        insta::assert_json_snapshot!(snapshot);
    }
}
