//! E2E scenario: tool call (function calling) — OpenAI and Anthropic formats.
//!
//! Covers:
//! - OpenAI `tool_calls` in response (`finish_reason: "tool_calls"`)
//! - OpenAI multi-turn: tool result sent back as `role: "tool"` message
//! - Anthropic `tool_use` content block in response
//! - Transform invariant: `arguments` is a JSON string on the OpenAI endpoint
//! - Tool definitions forwarded intact to the upstream

use serde_json::json;

use crate::assertions::assert_received_n;
use crate::harness::TestHarnessBuilder;
use crate::mocks::anthropic::{mock_messages_ok, mock_messages_with_tool_use};
use crate::mocks::openai::{mock_chat_after_tool, mock_chat_with_tool_calls};

/// Happy-path: the proxy returns tool_calls from the upstream with the correct
/// finish_reason and function name.
///
/// The mock returns an OpenAI response with `finish_reason: "tool_calls"` and a
/// single tool call for `get_weather`. The test verifies that the proxy
/// preserves the structure and delivers it to the client unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_openai_tool_call_in_response() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_with_tool_calls(&[("call_123", "get_weather", r#"{"city":"Paris"}"#)])
        .mount(openai_mock)
        .await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        },
                        "required": ["city"]
                    }
                }
            }]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let body: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(
        body["choices"][0]["finish_reason"], "tool_calls",
        "finish_reason must be 'tool_calls'"
    );

    let tool_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("expected tool_calls array in response");

    assert_eq!(tool_calls.len(), 1, "expected exactly one tool call");

    assert_eq!(
        tool_calls[0]["function"]["name"], "get_weather",
        "tool call function name must match"
    );

    let arguments_str = tool_calls[0]["function"]["arguments"]
        .as_str()
        .expect("arguments must be a JSON string, not an object");

    assert!(
        arguments_str.contains("Paris"),
        "arguments string must contain 'Paris', got: {arguments_str:?}"
    );
}

/// Multi-turn tool call: send a messages array that includes an assistant tool
/// call and a tool result message, then verify the final assistant content is
/// returned by the mock.
///
/// This exercises the `Role::Tool` → Anthropic `tool_result` content block
/// (on the OpenAI side) and ensures the proxy correctly marshals the full
/// conversation history to the upstream.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_openai_tool_result_round_trip() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_after_tool("The weather in Paris is sunny, 22°C.")
        .mount(openai_mock)
        .await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "22°C and sunny"
                }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        }
                    }
                }
            }]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let body: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(
        body["choices"][0]["message"]["content"], "The weather in Paris is sunny, 22°C.",
        "final assistant content must match the mock response"
    );
}

/// Anthropic tool_use in response: the `/api/v1/messages` endpoint must return
/// a `content` array with a `tool_use` block when the upstream responds with
/// `stop_reason: "tool_use"`.
///
/// The mock returns an Anthropic response with `content[0].type == "tool_use"`.
/// The proxy converts the internal `ProxiedResponse` back to Anthropic format
/// via `proxied_to_anthropic`, which re-emits the `tool_use` block with
/// `input` as a JSON object (not a string).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_anthropic_tool_use_in_response() {
    let harness = TestHarnessBuilder::new().with_anthropic().build().await;
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_messages_with_tool_use(
        "search_web",
        "toolu_001",
        json!({"query": "Rust programming"}),
    )
    .mount(anthropic_mock)
    .await;

    let resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Search for Rust programming resources"}],
            "max_tokens": 256,
            "tools": [{
                "name": "search_web",
                "description": "Search the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let body: serde_json::Value = resp.json().await.unwrap();

    // The Anthropic endpoint returns Anthropic-format responses.
    assert_eq!(body["type"], "message", "response type must be 'message'");
    assert_eq!(
        body["role"], "assistant",
        "response role must be 'assistant'"
    );

    // Find the tool_use block in the content array.
    let content = body["content"]
        .as_array()
        .expect("content must be an array");

    let tool_use_block = content
        .iter()
        .find(|block| block["type"] == "tool_use")
        .expect("response must contain a tool_use content block");

    assert_eq!(
        tool_use_block["name"], "search_web",
        "tool_use name must be 'search_web'"
    );

    // `input` must be a JSON object (not a string) in Anthropic format.
    assert!(
        tool_use_block["input"].is_object(),
        "tool_use input must be a JSON object in Anthropic response format, got: {:?}",
        tool_use_block["input"]
    );

    assert_eq!(
        tool_use_block["input"]["query"], "Rust programming",
        "tool_use input.query must match"
    );
}

/// Transform invariant: the proxy MUST return `arguments` as a JSON string
/// (not a parsed JSON object) on the OpenAI-compatible endpoint.
///
/// OpenAI's API contract specifies that `tool_calls[].function.arguments` is
/// always a string containing serialized JSON. The proxy must preserve this
/// even though the internal `ProxiedResponse` stores it as a `String`.
///
/// If this assertion fails (i.e., `arguments` is returned as an object), it
/// would indicate a regression in `proxied_to_openai` that breaks the OpenAI
/// wire contract.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_openai_tool_call_arguments_are_string() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // The mock upstream returns arguments as a JSON string (OpenAI wire format).
    mock_chat_with_tool_calls(&[("call_456", "get_location", r#"{"city":"London"}"#)])
        .mount(openai_mock)
        .await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Where is London?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_location",
                    "description": "Get location info",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        }
                    }
                }
            }]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let body: serde_json::Value = resp.json().await.unwrap();

    let tool_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("expected tool_calls array");

    assert!(!tool_calls.is_empty(), "expected at least one tool call");

    let arguments = &tool_calls[0]["function"]["arguments"];

    // CRITICAL: arguments must be a JSON string, not a parsed JSON object.
    // This matches the OpenAI wire format: clients expect to call
    // `JSON.parse(tool_calls[0].function.arguments)` on the client side.
    assert!(
        arguments.is_string(),
        "tool_calls[0].function.arguments must be a JSON string (not an object), \
         got type: {:?}, value: {arguments}",
        arguments.as_object().map(|_| "object").unwrap_or("other")
    );

    let arguments_str = arguments.as_str().unwrap();
    assert!(
        arguments_str.contains("London"),
        "arguments string must contain 'London', got: {arguments_str:?}"
    );
}

/// Tool definitions sent by the client must be forwarded to the upstream
/// unchanged. The proxy must include the `tools` array in the request body
/// it sends to the upstream LLM provider.
///
/// This test inspects the raw request body received by the wiremock server to
/// confirm the `tools` key is present, verifying no information is dropped
/// during the OpenAI → internal → upstream transform.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_tool_call_request_reaches_upstream() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_with_tool_calls(&[("call_789", "calculate", r#"{"expression":"2+2"}"#)])
        .mount(openai_mock)
        .await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Calculate 2+2"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            }]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    // The upstream mock must have received exactly one request.
    assert_received_n(openai_mock, 1).await;

    // Inspect the raw body that reached the upstream to confirm `tools` was forwarded.
    let received = openai_mock
        .received_requests()
        .await
        .expect("failed to fetch received requests");

    assert_eq!(received.len(), 1, "expected exactly one upstream request");

    let upstream_body: serde_json::Value =
        serde_json::from_slice(&received[0].body).expect("upstream request body is not valid JSON");

    assert!(
        upstream_body.get("tools").is_some(),
        "proxy must forward the 'tools' array to the upstream provider"
    );

    let tools = upstream_body["tools"]
        .as_array()
        .expect("tools must be an array");
    assert!(!tools.is_empty(), "tools array must not be empty");

    // Verify the tool definition was preserved (name survives the round-trip).
    let tool_names: Vec<&str> = tools
        .iter()
        .filter_map(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
        })
        .collect();

    assert!(
        tool_names.contains(&"calculate"),
        "upstream request must contain the 'calculate' tool definition, got: {tool_names:?}"
    );
}

/// Anthropic `stop_reason` is `"tool_use"` when the model wants to call a tool.
/// This test verifies the proxy correctly surfaces the stop_reason in the
/// Anthropic-format response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_anthropic_tool_use_stop_reason() {
    let harness = TestHarnessBuilder::new().with_anthropic().build().await;
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_messages_with_tool_use("lookup_price", "toolu_002", json!({"ticker": "AAPL"}))
        .mount(anthropic_mock)
        .await;

    let resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "What is the AAPL stock price?"}],
            "max_tokens": 256,
            "tools": [{
                "name": "lookup_price",
                "description": "Look up the current stock price",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"}
                    },
                    "required": ["ticker"]
                }
            }]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let body: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(
        body["stop_reason"], "tool_use",
        "stop_reason must be 'tool_use' when the model calls a tool"
    );

    // The mock upstream server must have been called exactly once.
    assert_received_n(anthropic_mock, 1).await;
}

/// Verify that an OpenAI response with multiple tool calls is preserved fully
/// — all tool calls are returned to the client.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_openai_multiple_tool_calls_preserved() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_with_tool_calls(&[
        ("call_a", "get_weather", r#"{"city":"Paris"}"#),
        ("call_b", "get_weather", r#"{"city":"London"}"#),
    ])
    .mount(openai_mock)
    .await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Weather in Paris and London?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        }
                    }
                }
            }]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    let body: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(
        body["choices"][0]["finish_reason"], "tool_calls",
        "finish_reason must be 'tool_calls'"
    );

    let tool_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("expected tool_calls array");

    assert_eq!(
        tool_calls.len(),
        2,
        "both tool calls must be forwarded to the client, got {}",
        tool_calls.len()
    );

    // Verify all arguments are strings (invariant must hold for all calls).
    for (i, tc) in tool_calls.iter().enumerate() {
        assert!(
            tc["function"]["arguments"].is_string(),
            "tool_calls[{i}].function.arguments must be a string"
        );
    }

    // Verify the call IDs are preserved.
    let ids: Vec<&str> = tool_calls
        .iter()
        .filter_map(|tc| tc["id"].as_str())
        .collect();
    assert!(ids.contains(&"call_a"), "call_a must be in the response");
    assert!(ids.contains(&"call_b"), "call_b must be in the response");
}

/// When an Anthropic request includes tool definitions, the upstream must
/// receive the `tools` array in Anthropic's native format (with `input_schema`,
/// not `parameters`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_anthropic_tool_definitions_forwarded_in_native_format() {
    let harness = TestHarnessBuilder::new().with_anthropic().build().await;
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    // Use a plain response mock — we just want to inspect the upstream request.
    mock_messages_ok("acknowledged").mount(anthropic_mock).await;

    let resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
            "tools": [{
                "name": "ping",
                "description": "Ping a host",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"}
                    },
                    "required": ["host"]
                }
            }]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 OK");

    // Confirm the upstream received exactly one request.
    assert_received_n(anthropic_mock, 1).await;

    let received = anthropic_mock
        .received_requests()
        .await
        .expect("failed to fetch received requests");

    let upstream_body: serde_json::Value =
        serde_json::from_slice(&received[0].body).expect("upstream request body is not valid JSON");

    assert!(
        upstream_body.get("tools").is_some(),
        "proxy must forward 'tools' to the Anthropic upstream"
    );

    let tools = upstream_body["tools"]
        .as_array()
        .expect("tools must be an array");

    assert!(!tools.is_empty(), "tools array must not be empty");

    // Anthropic native format uses `input_schema` (not `parameters`).
    let first_tool = &tools[0];
    assert_eq!(
        first_tool["name"], "ping",
        "tool name must be forwarded correctly"
    );
    assert!(
        first_tool.get("input_schema").is_some(),
        "Anthropic upstream must receive 'input_schema' (not 'parameters'), \
         got keys: {:?}",
        first_tool.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );
}
