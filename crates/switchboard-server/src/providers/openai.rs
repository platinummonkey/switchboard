//! OpenAI-compatible upstream provider.
//!
//! Used for OpenAI itself and any OpenAI-compatible API (Ollama, etc.).

use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use switchboard_common::errors::SwitchboardError;
use switchboard_common::types::{
    FunctionCall, ProxiedRequest, ProxiedResponse, Role, ToolCall, Usage,
};

use crate::config::provider::ProviderConfig;
use crate::key_pool::PooledKey;
use crate::routing::{BoxStream, UpstreamProvider};

// ── OpenAiProvider ────────────────────────────────────────────────────────────

/// Upstream provider for OpenAI and OpenAI-compatible APIs.
#[derive(Debug)]
pub struct OpenAiProvider {
    /// Short provider name (e.g. "openai" or "ollama").
    provider_name: String,
    /// Base URL for the API, e.g. "https://api.openai.com".
    base_url: String,
    /// Upstream request timeout (stored for future connection-level configuration).
    #[allow(dead_code)]
    timeout: Duration,
    /// Model names this provider can serve.
    models: Vec<String>,
    /// Shared reqwest client (reuse connection pool).
    client: reqwest::Client,
}

impl OpenAiProvider {
    /// Create a new provider from the server config's provider entry.
    pub fn new(name: impl Into<String>, config: &ProviderConfig) -> Result<Self, String> {
        let base_url = config
            .base_url
            .clone()
            .ok_or_else(|| format!("provider '{}': base_url is required", name.into()))?;

        let timeout = parse_duration(&config.timeout).unwrap_or(Duration::from_secs(300));

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| format!("failed to build reqwest client: {e}"))?;

        let provider_name = {
            // Re-derive name since we moved it above.
            config.api_format.clone()
        };

        Ok(Self {
            provider_name,
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout,
            models: config.models.clone(),
            client,
        })
    }

    /// Create a provider with an explicit name (for named registrations like "ollama").
    pub fn new_named(
        name: impl Into<String>,
        base_url: impl Into<String>,
        models: Vec<String>,
        timeout: Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("failed to build reqwest client");

        Self {
            provider_name: name.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            timeout,
            models,
            client,
        }
    }

    /// Build the request body for a non-streaming call.
    fn build_request_body(&self, request: &ProxiedRequest, stream: bool) -> serde_json::Value {
        let mut messages: Vec<serde_json::Value> = Vec::new();

        // Inject system prompt as a system message if present.
        if let Some(system) = &request.system {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system,
            }));
        }

        for msg in &request.messages {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            let mut m = serde_json::json!({
                "role": role,
                "content": msg.content.as_text(),
            });
            if let Some(id) = &msg.tool_call_id {
                m["tool_call_id"] = serde_json::Value::String(id.clone());
            }
            if let Some(calls) = &msg.tool_calls {
                let tc: Vec<_> = calls
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "id": c.id,
                            "type": c.call_type,
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            }
                        })
                    })
                    .collect();
                m["tool_calls"] = serde_json::Value::Array(tc);
            }
            messages.push(m);
        }

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "stream": stream,
        });

        if let Some(mt) = request.max_tokens {
            body["max_tokens"] = serde_json::Value::Number(mt.into());
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
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(tool_defs);
        }

        body
    }

    /// Parse an OpenAI API response body into a [`ProxiedResponse`].
    fn parse_response(
        body: &serde_json::Value,
        model: &str,
    ) -> Result<ProxiedResponse, SwitchboardError> {
        let content = body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let finish_reason = body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("finish_reason"))
            .and_then(|f| f.as_str())
            .map(|s| s.to_string());

        let tool_calls = body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("tool_calls"))
            .and_then(|tc| tc.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|tc| {
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
                        ToolCall {
                            id,
                            call_type,
                            function: FunctionCall { name, arguments },
                        }
                    })
                    .collect()
            });

        let usage = body.get("usage").map(|u| Usage {
            input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            output_tokens: u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
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
            tool_calls,
            finish_reason,
            usage,
        })
    }
}

#[async_trait]
impl UpstreamProvider for OpenAiProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn api_format(&self) -> &str {
        "openai"
    }

    fn supports_model(&self, model: &str) -> bool {
        self.models.iter().any(|m| m == model)
    }

    async fn send(
        &self,
        request: ProxiedRequest,
        key: &PooledKey,
    ) -> Result<ProxiedResponse, SwitchboardError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_request_body(&request, false);

        tracing::debug!(
            provider = self.name(),
            model = %request.model,
            url = %url,
            "sending non-streaming request to OpenAI-compatible provider"
        );

        let response = self
            .client
            .post(&url)
            .header(
                key.credentials.header_name.clone(),
                key.credentials.header_value.clone(),
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| SwitchboardError::Upstream(format!("request failed: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let err_body = response.text().await.unwrap_or_default();
            tracing::warn!(
                provider = self.name(),
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
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_request_body(&request, true);

        tracing::debug!(
            provider = self.name(),
            model = %request.model,
            url = %url,
            "sending streaming request to OpenAI-compatible provider"
        );

        let response = self
            .client
            .post(&url)
            .header(
                key.credentials.header_name.clone(),
                key.credentials.header_value.clone(),
            )
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

    async fn health_check(&self, key: &PooledKey) -> bool {
        let url = format!("{}/v1/models", self.base_url);
        match self
            .client
            .get(&url)
            .header(
                key.credentials.header_name.clone(),
                key.credentials.header_value.clone(),
            )
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(e) => {
                tracing::warn!(
                    provider = self.name(),
                    error = %e,
                    "health check failed"
                );
                false
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parse a human-readable duration string like "300s", "5m", "1h" into a
/// [`Duration`].  Returns `None` if the string cannot be parsed.
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
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer sk-test"),
                expires_at: None,
            },
            weight: 1.0,
            source: KeySource::Static,
            health: KeyHealth::default(),
        }
    }

    fn make_provider(base_url: &str) -> OpenAiProvider {
        OpenAiProvider::new_named(
            "openai",
            base_url,
            vec!["gpt-4o".into(), "gpt-4o-mini".into()],
            Duration::from_secs(30),
        )
    }

    #[test]
    fn test_provider_name_and_format() {
        let p = make_provider("https://api.openai.com");
        assert_eq!(p.name(), "openai");
        assert_eq!(p.api_format(), "openai");
    }

    #[test]
    fn test_supports_model_known() {
        let p = make_provider("https://api.openai.com");
        assert!(p.supports_model("gpt-4o"));
        assert!(p.supports_model("gpt-4o-mini"));
    }

    #[test]
    fn test_supports_model_unknown() {
        let p = make_provider("https://api.openai.com");
        assert!(!p.supports_model("claude-sonnet-4-20250514"));
    }

    #[test]
    fn test_build_request_body_basic() {
        let p = make_provider("https://api.openai.com");
        let req = ProxiedRequest {
            model: "gpt-4o".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("hello".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(512),
            temperature: Some(0.7),
            top_p: None,
            system: None,
            extra: Default::default(),
        };
        let body = p.build_request_body(&req, false);
        assert_eq!(body["model"], "gpt-4o");
        assert!(!body["stream"].as_bool().unwrap());
        assert_eq!(body["max_tokens"], 512);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[test]
    fn test_build_request_body_with_system() {
        let p = make_provider("https://api.openai.com");
        let req = ProxiedRequest {
            model: "gpt-4o".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("hi".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            system: Some("Be helpful".into()),
            extra: Default::default(),
        };
        let body = p.build_request_body(&req, false);
        // System message should be prepended.
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "Be helpful");
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn test_parse_response_basic() {
        let json = serde_json::json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });
        let resp = OpenAiProvider::parse_response(&json, "gpt-4o").unwrap();
        assert_eq!(resp.content, "Hello!");
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        let u = resp.usage.unwrap();
        assert_eq!(u.input_tokens, 10);
        assert_eq!(u.output_tokens, 5);
    }

    #[test]
    fn test_parse_response_empty_choices() {
        let json = serde_json::json!({
            "choices": [],
            "usage": null
        });
        let resp = OpenAiProvider::parse_response(&json, "gpt-4o").unwrap();
        assert_eq!(resp.content, "");
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("300s"), Some(Duration::from_secs(300)));
        assert_eq!(parse_duration("5m"), Some(Duration::from_secs(300)));
        assert_eq!(parse_duration("1h"), Some(Duration::from_secs(3600)));
        assert_eq!(parse_duration("60"), Some(Duration::from_secs(60)));
        assert_eq!(parse_duration("invalid"), None);
    }

    #[tokio::test]
    async fn test_send_non_streaming_mock() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let key = make_key();

        let req = ProxiedRequest {
            model: "gpt-4o".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("Hello".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(256),
            temperature: None,
            top_p: None,
            system: None,
            extra: Default::default(),
        };

        let resp = provider.send(req, &key).await.unwrap();
        assert_eq!(resp.content, "Hi there!");
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 3);
    }

    #[tokio::test]
    async fn test_send_non_streaming_upstream_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("Rate limited"))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let key = make_key();

        let req = ProxiedRequest {
            model: "gpt-4o".into(),
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
        assert!(result.unwrap_err().to_string().contains("429"));
    }

    #[tokio::test]
    async fn test_health_check_success() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"object": "list", "data": []})),
            )
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let key = make_key();
        assert!(provider.health_check(&key).await);
    }

    #[tokio::test]
    async fn test_health_check_failure() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let key = make_key();
        assert!(!provider.health_check(&key).await);
    }
}
