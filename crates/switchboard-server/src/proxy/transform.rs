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

    let (message, finish_reason) = if let Some(tool_calls) = &resp.tool_calls {
        // When tool calls are present: content is null, finish_reason is "tool_calls".
        let tc_json: Vec<serde_json::Value> = tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "type": tc.call_type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            })
            .collect();
        let msg = serde_json::json!({
            "role": "assistant",
            "content": serde_json::Value::Null,
            "tool_calls": tc_json,
        });
        (msg, "tool_calls")
    } else {
        let msg = serde_json::json!({
            "role": "assistant",
            "content": resp.content,
        });
        let fr = resp.finish_reason.as_deref().unwrap_or("stop");
        (msg, fr)
    };

    serde_json::json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4().simple()),
        "object": "chat.completion",
        "model": resp.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
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
///
/// Handles the special case of `role: "user"` with `type: "tool_result"` content
/// blocks — these are converted to `Role::Tool` messages with `tool_call_id`.
fn parse_anthropic_message(msg: &serde_json::Value, index: usize) -> Result<Message, ProxyError> {
    let role_str = msg.get("role").and_then(|v| v.as_str()).ok_or_else(|| {
        ProxyError::InvalidRequest(format!("messages[{index}]: missing role field"))
    })?;

    // Check for tool_result content blocks (sent back as role: "user").
    if role_str == "user" {
        if let Some(serde_json::Value::Array(parts)) = msg.get("content") {
            // If the first content block is of type "tool_result", treat the
            // whole message as a Tool role message.
            let first_is_tool_result = parts
                .first()
                .is_some_and(|p| p.get("type").and_then(|t| t.as_str()) == Some("tool_result"));
            if first_is_tool_result {
                // Use the first tool_result block as the canonical one.
                let block = &parts[0];
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let content_text = match block.get("content") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Array(arr)) => arr
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
                        .join("\n"),
                    _ => String::new(),
                };
                return Ok(Message {
                    role: Role::Tool,
                    content: MessageContent::Text(content_text),
                    tool_call_id: Some(tool_use_id),
                    tool_calls: None,
                });
            }
        }
    }

    let role = match role_str {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        other => {
            return Err(ProxyError::InvalidRequest(format!(
                "messages[{index}]: unknown Anthropic role '{other}'"
            )));
        }
    };

    // For assistant messages: extract tool_calls from tool_use content blocks
    // and text from text content blocks.
    if role == Role::Assistant {
        if let Some(serde_json::Value::Array(parts)) = msg.get("content") {
            let has_tool_use = parts
                .iter()
                .any(|p| p.get("type").and_then(|t| t.as_str()) == Some("tool_use"));
            if has_tool_use {
                let text_content: String = parts
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

                let tool_calls: Vec<ToolCall> = parts
                    .iter()
                    .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
                    .map(|p| {
                        let id = p
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = p
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = p
                            .get("input")
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "{}".to_string());
                        ToolCall {
                            id,
                            call_type: "function".to_string(),
                            function: FunctionCall { name, arguments },
                        }
                    })
                    .collect();

                return Ok(Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(text_content),
                    tool_call_id: None,
                    tool_calls: Some(tool_calls),
                });
            }
        }
    }

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

    // Build content array: text block + optional tool_use blocks.
    let mut content_blocks: Vec<serde_json::Value> = Vec::new();
    if !resp.content.is_empty() {
        content_blocks.push(serde_json::json!({"type": "text", "text": resp.content}));
    }
    if let Some(tool_calls) = &resp.tool_calls {
        for tc in tool_calls {
            // Parse arguments back to a JSON object for Anthropic's `input` field.
            let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            content_blocks.push(serde_json::json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": input,
            }));
        }
    }

    // If there are no content blocks at all, emit an empty text block to
    // satisfy Anthropic's schema requirement.
    if content_blocks.is_empty() {
        content_blocks.push(serde_json::json!({"type": "text", "text": ""}));
    }

    let stop_reason = if resp.tool_calls.is_some() {
        "tool_use"
    } else {
        resp.finish_reason.as_deref().unwrap_or("end_turn")
    };

    serde_json::json!({
        "id": format!("msg_{}", Uuid::new_v4().simple()),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": resp.model,
        "stop_reason": stop_reason,
        "usage": usage,
    })
}

// ── Bedrock → internal ───────────────────────────────────────────────────────

/// Serialize a [`ProxiedRequest`] to Amazon Bedrock Converse API JSON format.
///
/// This is the canonical helper used by the test suite; the `BedrockProvider`
/// itself builds its own request body inline.
pub fn proxied_to_bedrock_converse(req: &ProxiedRequest) -> serde_json::Value {
    let system_text = req.system.clone().or_else(|| {
        req.messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.as_text())
    });

    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|msg| {
            let role = match msg.role {
                Role::User | Role::Tool => "user",
                Role::Assistant => "assistant",
                Role::System => unreachable!("filtered above"),
            };
            serde_json::json!({
                "role": role,
                "content": [{"text": msg.content.as_text()}],
            })
        })
        .collect();

    let max_tokens = req.max_tokens.unwrap_or(4096);
    let mut inference_config = serde_json::json!({"maxTokens": max_tokens});
    if let Some(temp) = req.temperature {
        inference_config["temperature"] = serde_json::json!(temp);
    }
    if let Some(tp) = req.top_p {
        inference_config["topP"] = serde_json::json!(tp);
    }

    let mut body = serde_json::json!({
        "messages": messages,
        "inferenceConfig": inference_config,
    });

    if let Some(sys) = system_text {
        body["system"] = serde_json::json!([{"text": sys}]);
    }

    body
}

/// Parse a Bedrock Converse API response body into a [`ProxiedResponse`].
pub fn bedrock_converse_to_proxied(body: &serde_json::Value) -> ProxiedResponse {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let content = body
        .get("output")
        .and_then(|o| o.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
        .and_then(|arr| {
            arr.iter().find_map(|block| {
                block
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
        })
        .unwrap_or_default();

    let finish_reason = body
        .get("stopReason")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let usage = body.get("usage").map(|u| switchboard_common::types::Usage {
        input_tokens: u.get("inputTokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        output_tokens: u.get("outputTokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        ..Default::default()
    });

    ProxiedResponse {
        model,
        content,
        tool_calls: None,
        finish_reason,
        usage,
    }
}

// ── Vertex → internal ────────────────────────────────────────────────────────

/// Serialize a [`ProxiedRequest`] to Google Vertex AI (Gemini) `generateContent`
/// request JSON format.
pub fn proxied_to_vertex_gemini(req: &ProxiedRequest) -> serde_json::Value {
    let system_text = req.system.clone().or_else(|| {
        req.messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.as_text())
    });

    let contents: Vec<serde_json::Value> = req
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|msg| {
            let role = match msg.role {
                Role::User | Role::Tool => "user",
                Role::Assistant => "model",
                Role::System => unreachable!("filtered above"),
            };
            serde_json::json!({
                "role": role,
                "parts": [{"text": msg.content.as_text()}],
            })
        })
        .collect();

    let max_output_tokens = req.max_tokens.unwrap_or(4096);
    let mut generation_config = serde_json::json!({"maxOutputTokens": max_output_tokens});
    if let Some(temp) = req.temperature {
        generation_config["temperature"] = serde_json::json!(temp);
    }
    if let Some(tp) = req.top_p {
        generation_config["topP"] = serde_json::json!(tp);
    }

    let mut body = serde_json::json!({
        "contents": contents,
        "generationConfig": generation_config,
    });

    if let Some(sys) = system_text {
        body["systemInstruction"] = serde_json::json!({"parts": [{"text": sys}]});
    }

    body
}

/// Parse a Vertex AI `generateContent` response body into a [`ProxiedResponse`].
pub fn vertex_gemini_to_proxied(body: &serde_json::Value) -> ProxiedResponse {
    let model = body
        .get("modelVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let content = body
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .and_then(|arr| {
            arr.iter().find_map(|part| {
                part.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
        })
        .unwrap_or_default();

    let finish_reason = body
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finishReason"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let usage = body
        .get("usageMetadata")
        .map(|u| switchboard_common::types::Usage {
            input_tokens: u
                .get("promptTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            output_tokens: u
                .get("candidatesTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            ..Default::default()
        });

    ProxiedResponse {
        model,
        content,
        tool_calls: None,
        finish_reason,
        usage,
    }
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

    // ── proxied_to_bedrock_converse ──────────────────────────────────────────

    #[test]
    fn test_proxied_to_bedrock_converse_basic() {
        let req = ProxiedRequest {
            model: "anthropic.claude-3-sonnet".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Hello!".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(256),
            temperature: Some(0.7),
            top_p: None,
            system: None,
            extra: Default::default(),
        };
        let body = proxied_to_bedrock_converse(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["text"], "Hello!");
        assert_eq!(body["inferenceConfig"]["maxTokens"], 256);
        assert!(
            (body["inferenceConfig"]["temperature"].as_f64().unwrap() - 0.7).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_proxied_to_bedrock_converse_with_system() {
        let req = ProxiedRequest {
            model: "anthropic.claude-3-sonnet".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Hi".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            system: Some("Be helpful.".into()),
            extra: Default::default(),
        };
        let body = proxied_to_bedrock_converse(&req);
        assert_eq!(body["system"][0]["text"], "Be helpful.");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_bedrock_converse_to_proxied_basic() {
        let json = serde_json::json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello from Bedrock!"}]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "stopReason": "end_turn"
        });
        let resp = bedrock_converse_to_proxied(&json);
        assert_eq!(resp.content, "Hello from Bedrock!");
        assert_eq!(resp.finish_reason.as_deref(), Some("end_turn"));
        let u = resp.usage.unwrap();
        assert_eq!(u.input_tokens, 10);
        assert_eq!(u.output_tokens, 5);
    }

    #[test]
    fn test_bedrock_converse_to_proxied_empty() {
        let json = serde_json::json!({});
        let resp = bedrock_converse_to_proxied(&json);
        assert_eq!(resp.content, "");
        assert!(resp.finish_reason.is_none());
        assert!(resp.usage.is_none());
    }

    // ── proxied_to_vertex_gemini ─────────────────────────────────────────────

    #[test]
    fn test_proxied_to_vertex_gemini_basic() {
        let req = ProxiedRequest {
            model: "gemini-1.5-pro".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Hello Gemini!".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(512),
            temperature: Some(0.5),
            top_p: None,
            system: None,
            extra: Default::default(),
        };
        let body = proxied_to_vertex_gemini(&req);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Hello Gemini!");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 512);
        assert!(
            (body["generationConfig"]["temperature"].as_f64().unwrap() - 0.5).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_proxied_to_vertex_gemini_assistant_role() {
        let req = ProxiedRequest {
            model: "gemini-1.5-pro".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: MessageContent::Text("Hi".into()),
                    tool_call_id: None,
                    tool_calls: None,
                },
                Message {
                    role: Role::Assistant,
                    content: MessageContent::Text("Hello!".into()),
                    tool_call_id: None,
                    tool_calls: None,
                },
            ],
            tools: None,
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            system: None,
            extra: Default::default(),
        };
        let body = proxied_to_vertex_gemini(&req);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents[1]["role"], "model");
    }

    #[test]
    fn test_proxied_to_vertex_gemini_with_system() {
        let req = ProxiedRequest {
            model: "gemini-1.5-pro".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("hi".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            system: Some("Be concise.".into()),
            extra: Default::default(),
        };
        let body = proxied_to_vertex_gemini(&req);
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be concise.");
    }

    #[test]
    fn test_vertex_gemini_to_proxied_basic() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20
            }
        });
        let resp = vertex_gemini_to_proxied(&json);
        assert_eq!(resp.content, "Hello from Gemini!");
        assert_eq!(resp.finish_reason.as_deref(), Some("STOP"));
        let u = resp.usage.unwrap();
        assert_eq!(u.input_tokens, 10);
        assert_eq!(u.output_tokens, 20);
    }

    #[test]
    fn test_vertex_gemini_to_proxied_no_candidates() {
        let json = serde_json::json!({"candidates": []});
        let resp = vertex_gemini_to_proxied(&json);
        assert_eq!(resp.content, "");
        assert!(resp.finish_reason.is_none());
    }

    // ── Insta snapshot tests for new transform functions ─────────────────────

    #[test]
    fn test_snapshot_proxied_to_bedrock_converse() {
        let req = ProxiedRequest {
            model: "anthropic.claude-3-sonnet".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("What is 2+2?".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(256),
            temperature: Some(0.7),
            top_p: None,
            system: Some("Be concise.".into()),
            extra: Default::default(),
        };
        let body = proxied_to_bedrock_converse(&req);
        insta::assert_json_snapshot!(body);
    }

    #[test]
    fn test_snapshot_bedrock_converse_to_proxied() {
        let json = serde_json::json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "4"}]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 2},
            "stopReason": "end_turn"
        });
        let resp = bedrock_converse_to_proxied(&json);
        let snapshot = serde_json::json!({
            "content": resp.content,
            "finish_reason": resp.finish_reason,
            "input_tokens": resp.usage.as_ref().map(|u| u.input_tokens),
            "output_tokens": resp.usage.as_ref().map(|u| u.output_tokens),
        });
        insta::assert_json_snapshot!(snapshot);
    }

    #[test]
    fn test_snapshot_proxied_to_vertex_gemini() {
        let req = ProxiedRequest {
            model: "gemini-1.5-pro".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("What is 2+2?".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(128),
            temperature: Some(0.5),
            top_p: None,
            system: Some("Be concise.".into()),
            extra: Default::default(),
        };
        let body = proxied_to_vertex_gemini(&req);
        insta::assert_json_snapshot!(body);
    }

    #[test]
    fn test_snapshot_vertex_gemini_to_proxied() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "4"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 2
            }
        });
        let resp = vertex_gemini_to_proxied(&json);
        let snapshot = serde_json::json!({
            "content": resp.content,
            "finish_reason": resp.finish_reason,
            "input_tokens": resp.usage.as_ref().map(|u| u.input_tokens),
            "output_tokens": resp.usage.as_ref().map(|u| u.output_tokens),
        });
        insta::assert_json_snapshot!(snapshot);
    }

    // ── Tool call round-trip tests ────────────────────────────────────────────

    /// A `ProxiedRequest` with tools serialized via `AnthropicProvider::build_request_body`
    /// is tested in `anthropic.rs`; here we verify the `transform` layer directly.

    #[test]
    fn test_anthropic_to_proxied_extracts_tool_use() {
        // Anthropic response with a single tool_use block.
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "calculator",
                        "input": {"expression": "2+2"}
                    }
                ]
            }],
            "max_tokens": 1024
        });
        let req = anthropic_to_proxied(&body).unwrap();
        assert_eq!(req.messages.len(), 1);
        let msg = &req.messages[0];
        assert_eq!(msg.role, Role::Assistant);
        let tc = msg.tool_calls.as_ref().expect("tool_calls should be Some");
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "toolu_abc");
        assert_eq!(tc[0].function.name, "calculator");
        // The arguments should be the JSON representation of the input object.
        let args: serde_json::Value = serde_json::from_str(&tc[0].function.arguments).unwrap();
        assert_eq!(args["expression"], "2+2");
    }

    #[test]
    fn test_anthropic_to_proxied_mixed_content() {
        // Anthropic response with text + tool_use blocks.
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll use the calculator."},
                    {
                        "type": "tool_use",
                        "id": "toolu_xyz",
                        "name": "calculator",
                        "input": {"expression": "3*3"}
                    }
                ]
            }],
            "max_tokens": 1024
        });
        let req = anthropic_to_proxied(&body).unwrap();
        let msg = &req.messages[0];
        // Text content should only contain text blocks.
        assert_eq!(msg.content.as_text(), "I'll use the calculator.");
        // Tool calls should have the tool_use block.
        let tc = msg.tool_calls.as_ref().expect("tool_calls should be Some");
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "toolu_xyz");
    }

    #[test]
    fn test_proxied_to_openai_with_tool_calls() {
        // A ProxiedResponse with tool_calls should produce OpenAI format with
        // null content and finish_reason "tool_calls".
        let resp = ProxiedResponse {
            model: "gpt-4o".into(),
            content: String::new(),
            tool_calls: Some(vec![ToolCall {
                id: "call_123".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"NYC"}"#.into(),
                },
            }]),
            finish_reason: Some("tool_calls".into()),
            usage: None,
        };
        let json = proxied_to_openai(&resp);
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
        assert!(json["choices"][0]["message"]["content"].is_null());
        let tcs = &json["choices"][0]["message"]["tool_calls"];
        assert!(tcs.is_array());
        assert_eq!(tcs[0]["id"], "call_123");
        assert_eq!(tcs[0]["function"]["name"], "get_weather");
        assert_eq!(tcs[0]["function"]["arguments"], r#"{"city":"NYC"}"#);
    }

    #[test]
    fn test_openai_to_proxied_tool_role_message() {
        // OpenAI request with a tool role message (result sent back after tool call).
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": r#"{"expression":"2+2"}"#}
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc",
                    "content": "4"
                }
            ]
        });
        let req = openai_to_proxied(&body).unwrap();
        assert_eq!(req.messages.len(), 3);
        // Last message should be Role::Tool with tool_call_id.
        let tool_msg = &req.messages[2];
        assert_eq!(tool_msg.role, Role::Tool);
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_abc"));
        assert_eq!(tool_msg.content.as_text(), "4");
        // Assistant message should have tool_calls.
        let asst_msg = &req.messages[1];
        let tc = asst_msg.tool_calls.as_ref().expect("assistant tool_calls");
        assert_eq!(tc[0].id, "call_abc");
        assert_eq!(tc[0].function.name, "calculator");
    }

    #[test]
    fn test_anthropic_request_with_tool_result() {
        // Anthropic request with a tool_result content block (user turn after tool use).
        let body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "calculator",
                        "input": {"expression": "2+2"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "4"
                    }]
                }
            ],
            "max_tokens": 1024
        });
        let req = anthropic_to_proxied(&body).unwrap();
        assert_eq!(req.messages.len(), 3);
        // Third message should be converted to Role::Tool.
        let tool_msg = &req.messages[2];
        assert_eq!(tool_msg.role, Role::Tool);
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("toolu_abc"));
        assert_eq!(tool_msg.content.as_text(), "4");
    }
}
