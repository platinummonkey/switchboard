//! Anthropic Messages API upstream provider.

use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use switchboard_common::errors::SwitchboardError;
use switchboard_common::types::{ProxiedRequest, ProxiedResponse, Role, Usage};

use crate::config::provider::ProviderConfig;
use crate::key_pool::PooledKey;
use crate::routing::{BoxStream, UpstreamProvider};

/// Anthropic Messages API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Default base URL for the Anthropic API.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Default max_tokens when not specified (Anthropic requires it).
const DEFAULT_MAX_TOKENS: u32 = 4096;

// ── AnthropicProvider ─────────────────────────────────────────────────────────

/// Upstream provider for the native Anthropic Messages API.
#[derive(Debug)]
pub struct AnthropicProvider {
    /// Base URL for the API.
    base_url: String,
    /// Per-request timeout (stored for future connection-level configuration).
    #[allow(dead_code)]
    timeout: Duration,
    /// Model IDs served by this provider.
    models: Vec<String>,
    /// Shared reqwest client.
    client: reqwest::Client,
}

impl AnthropicProvider {
    /// Create a new provider from the server config's provider entry.
    pub fn new(_name: &str, config: &ProviderConfig) -> Result<Self, String> {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

        let timeout = parse_duration(&config.timeout).unwrap_or(Duration::from_secs(300));

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(config.max_concurrent as usize)
            .tcp_keepalive(Duration::from_secs(90))
            .build()
            .map_err(|e| format!("failed to build reqwest client: {e}"))?;

        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout,
            models: config.models.clone(),
            client,
        })
    }

    /// Create a provider with explicit parameters (useful for tests).
    pub fn new_with_base_url(
        base_url: impl Into<String>,
        models: Vec<String>,
        timeout: Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(64)
            .tcp_keepalive(Duration::from_secs(90))
            .build()
            .expect("failed to build reqwest client");

        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            timeout,
            models,
            client,
        }
    }

    /// Build the Anthropic Messages API request body.
    fn build_request_body(&self, request: &ProxiedRequest, stream: bool) -> serde_json::Value {
        // Collect the system prompt: prefer explicit `system` field, else
        // extract from a leading Role::System message.
        let system = request.system.clone().or_else(|| {
            request
                .messages
                .iter()
                .find(|m| m.role == Role::System)
                .map(|m| m.content.as_text())
        });

        // Filter out system messages (they go into the top-level `system` field).
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|msg| {
                let role = match msg.role {
                    Role::User | Role::Tool => "user",
                    Role::Assistant => "assistant",
                    Role::System => unreachable!("system messages filtered above"),
                };
                serde_json::json!({
                    "role": role,
                    "content": msg.content.as_text(),
                })
            })
            .collect();

        let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        });

        if let Some(sys) = system {
            body["system"] = serde_json::Value::String(sys);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tp) = request.top_p {
            body["top_p"] = serde_json::json!(tp);
        }
        if let Some(tools) = &request.tools {
            let tool_defs: Vec<_> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(tool_defs);
        }

        body
    }

    /// Parse an Anthropic Messages API response body into a [`ProxiedResponse`].
    fn parse_response(
        body: &serde_json::Value,
        model: &str,
    ) -> Result<ProxiedResponse, SwitchboardError> {
        // Extract text content from content array.
        let content = body
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| {
                arr.iter().find_map(|block| {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                        block
                            .get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_default();

        let finish_reason = body
            .get("stop_reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let usage = body.get("usage").map(|u| Usage {
            input_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            output_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            ..Default::default()
        });

        let response_model = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(model)
            .to_string();

        Ok(ProxiedResponse {
            model: response_model,
            content,
            tool_calls: None, // Tool call parsing for Anthropic is more complex; deferred.
            finish_reason,
            usage,
        })
    }
}

#[async_trait]
impl UpstreamProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn api_format(&self) -> &str {
        "anthropic"
    }

    fn supports_model(&self, model: &str) -> bool {
        self.models.iter().any(|m| m == model)
    }

    async fn send(
        &self,
        request: ProxiedRequest,
        key: &PooledKey,
    ) -> Result<ProxiedResponse, SwitchboardError> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = self.build_request_body(&request, false);

        tracing::debug!(
            provider = "anthropic",
            model = %request.model,
            url = %url,
            "sending non-streaming request to Anthropic"
        );

        let response = self
            .client
            .post(&url)
            .header(
                key.credentials.header_name.clone(),
                key.credentials.header_value.clone(),
            )
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| SwitchboardError::Upstream(format!("request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let err_body = response.text().await.unwrap_or_default();
            tracing::warn!(
                provider = "anthropic",
                status = %status,
                body = %err_body,
                "upstream returned error status"
            );
            return Err(SwitchboardError::Upstream(format!(
                "upstream HTTP {status}: {err_body}"
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| SwitchboardError::Upstream(format!("failed to parse response: {e}")))?;

        Self::parse_response(&json, &request.model)
    }

    async fn send_streaming(
        &self,
        request: ProxiedRequest,
        key: &PooledKey,
    ) -> Result<BoxStream, SwitchboardError> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = self.build_request_body(&request, true);

        tracing::debug!(
            provider = "anthropic",
            model = %request.model,
            url = %url,
            "sending streaming request to Anthropic"
        );

        let response = self
            .client
            .post(&url)
            .header(
                key.credentials.header_name.clone(),
                key.credentials.header_value.clone(),
            )
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| SwitchboardError::Upstream(format!("streaming request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let err_body = response.text().await.unwrap_or_default();
            return Err(SwitchboardError::Upstream(format!(
                "upstream HTTP {status}: {err_body}"
            )));
        }

        let byte_stream = response
            .bytes_stream()
            .map(|r| r.map_err(|e| SwitchboardError::Upstream(e.to_string())));

        Ok(Box::pin(byte_stream))
    }

    async fn health_check(&self, _key: &PooledKey) -> bool {
        // Anthropic has no public health endpoint; always report healthy.
        true
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if let Some(n) = s.strip_suffix('s') {
        n.trim().parse::<u64>().ok().map(Duration::from_secs)
    } else if let Some(n) = s.strip_suffix('m') {
        n.trim()
            .parse::<u64>()
            .ok()
            .map(|v| Duration::from_secs(v * 60))
    } else if let Some(n) = s.strip_suffix('h') {
        n.trim()
            .parse::<u64>()
            .ok()
            .map(|v| Duration::from_secs(v * 3600))
    } else {
        s.parse::<u64>().ok().map(Duration::from_secs)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use http::{HeaderName, HeaderValue};

    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::provider::KeySource;
    use crate::key_pool::{KeyHealth, PooledKey};

    fn make_key() -> PooledKey {
        PooledKey {
            id: "test-key".into(),
            credentials: UpstreamCredentials {
                header_name: HeaderName::from_static("x-api-key"),
                header_value: HeaderValue::from_static("sk-ant-test"),
                expires_at: None,
            },
            weight: 1.0,
            source: KeySource::Static,
            health: KeyHealth::default(),
        }
    }

    fn make_provider(base_url: &str) -> AnthropicProvider {
        AnthropicProvider::new_with_base_url(
            base_url,
            vec![
                "claude-sonnet-4-20250514".into(),
                "claude-opus-4-20250514".into(),
            ],
            Duration::from_secs(30),
        )
    }

    #[test]
    fn test_provider_name_and_format() {
        let p = make_provider("https://api.anthropic.com");
        assert_eq!(p.name(), "anthropic");
        assert_eq!(p.api_format(), "anthropic");
    }

    #[test]
    fn test_supports_model_known() {
        let p = make_provider("https://api.anthropic.com");
        assert!(p.supports_model("claude-sonnet-4-20250514"));
        assert!(p.supports_model("claude-opus-4-20250514"));
    }

    #[test]
    fn test_supports_model_unknown() {
        let p = make_provider("https://api.anthropic.com");
        assert!(!p.supports_model("gpt-4o"));
    }

    #[test]
    fn test_build_request_body_basic() {
        let p = make_provider("https://api.anthropic.com");
        let req = ProxiedRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("Hello!".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(1024),
            temperature: Some(0.5),
            top_p: None,
            system: None,
            extra: Default::default(),
        };
        let body = p.build_request_body(&req, false);
        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["max_tokens"], 1024);
        assert!(!body["stream"].as_bool().unwrap());
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Hello!");
    }

    #[test]
    fn test_build_request_body_default_max_tokens() {
        let p = make_provider("https://api.anthropic.com");
        let req = ProxiedRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("hi".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: None, // should default to DEFAULT_MAX_TOKENS
            temperature: None,
            top_p: None,
            system: None,
            extra: Default::default(),
        };
        let body = p.build_request_body(&req, false);
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn test_build_request_body_with_system() {
        let p = make_provider("https://api.anthropic.com");
        let req = ProxiedRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("hi".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(256),
            temperature: None,
            top_p: None,
            system: Some("You are a pirate.".into()),
            extra: Default::default(),
        };
        let body = p.build_request_body(&req, false);
        assert_eq!(body["system"], "You are a pirate.");
        // System messages should not be in the messages array.
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_build_request_body_filters_system_messages() {
        let p = make_provider("https://api.anthropic.com");
        let req = ProxiedRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![
                switchboard_common::types::Message {
                    role: Role::System,
                    content: switchboard_common::types::MessageContent::Text(
                        "You are helpful.".into(),
                    ),
                    tool_call_id: None,
                    tool_calls: None,
                },
                switchboard_common::types::Message {
                    role: Role::User,
                    content: switchboard_common::types::MessageContent::Text("Hello".into()),
                    tool_call_id: None,
                    tool_calls: None,
                },
            ],
            tools: None,
            stream: false,
            max_tokens: Some(256),
            temperature: None,
            top_p: None,
            system: None,
            extra: Default::default(),
        };
        let body = p.build_request_body(&req, false);
        // System message extracted to top-level.
        assert_eq!(body["system"], "You are helpful.");
        // Only the user message remains.
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn test_parse_response_basic() {
        let json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "I can help!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 15, "output_tokens": 8}
        });
        let resp = AnthropicProvider::parse_response(&json, "claude-sonnet-4-20250514").unwrap();
        assert_eq!(resp.content, "I can help!");
        assert_eq!(resp.model, "claude-sonnet-4-20250514");
        assert_eq!(resp.finish_reason.as_deref(), Some("end_turn"));
        let u = resp.usage.unwrap();
        assert_eq!(u.input_tokens, 15);
        assert_eq!(u.output_tokens, 8);
    }

    #[test]
    fn test_parse_response_multiple_content_blocks() {
        let json = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 42."}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
        });
        let resp = AnthropicProvider::parse_response(&json, "claude-sonnet-4-20250514").unwrap();
        // Should extract only text blocks.
        assert_eq!(resp.content, "The answer is 42.");
    }

    #[tokio::test]
    async fn test_health_check_always_true() {
        let p = make_provider("https://api.anthropic.com");
        let key = make_key();
        assert!(p.health_check(&key).await);
    }

    #[tokio::test]
    async fn test_send_non_streaming_mock() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-ant-test"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Anthropic!"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 6}
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let key = make_key();

        let req = ProxiedRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("Hello!".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(512),
            temperature: None,
            top_p: None,
            system: None,
            extra: Default::default(),
        };

        let resp = provider.send(req, &key).await.unwrap();
        assert_eq!(resp.content, "Hello from Anthropic!");
        assert_eq!(resp.finish_reason.as_deref(), Some("end_turn"));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 6);
    }

    #[tokio::test]
    async fn test_send_non_streaming_error_response() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "type": "error",
                "error": {"type": "authentication_error", "message": "invalid api key"}
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let key = make_key();

        let req = ProxiedRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("hello".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            system: None,
            extra: Default::default(),
        };

        let result = provider.send(req, &key).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("401"));
    }
}
