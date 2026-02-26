//! Google Vertex AI (Gemini) upstream provider.
//!
//! Uses the Vertex AI generateContent / streamGenerateContent endpoints.
//!
//! # Authentication
//!
//! The [`PooledKey`] carries an OAuth2 bearer token in the standard
//! `Authorization: Bearer <token>` header.
//!
//! # Endpoints
//!
//! Non-streaming:
//! `POST https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:generateContent`
//!
//! Streaming (SSE):
//! `POST https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:streamGenerateContent?alt=sse`

use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use switchboard_common::errors::SwitchboardError;
use switchboard_common::types::{ProxiedRequest, ProxiedResponse, Role, Usage};

use crate::config::provider::ProviderConfig;
use crate::key_pool::PooledKey;
use crate::routing::{BoxStream, UpstreamProvider};

// ── Constants ─────────────────────────────────────────────────────────────────

const DEFAULT_REGION: &str = "us-central1";
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

// ── VertexProvider ────────────────────────────────────────────────────────────

/// Upstream provider for Google Vertex AI (Gemini models).
#[derive(Debug)]
pub struct VertexProvider {
    /// GCP region, e.g. `"us-central1"`.
    region: String,
    /// GCP project ID.
    project_id: String,
    /// Model IDs served by this provider.
    models: Vec<String>,
    /// Shared reqwest client.
    client: reqwest::Client,
}

impl VertexProvider {
    /// Create a provider from the server config's provider entry.
    pub fn new(_name: &str, config: &ProviderConfig) -> Result<Self, String> {
        let region = config
            .region
            .clone()
            .unwrap_or_else(|| DEFAULT_REGION.to_string());

        let project_id = config
            .project_id
            .clone()
            .ok_or_else(|| "Vertex provider requires project_id in config".to_string())?;

        let timeout = parse_duration(&config.timeout).unwrap_or(Duration::from_secs(300));

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(config.max_concurrent as usize)
            .tcp_keepalive(Duration::from_secs(90))
            .build()
            .map_err(|e| format!("failed to build reqwest client: {e}"))?;

        Ok(Self {
            region,
            project_id,
            models: config.models.clone(),
            client,
        })
    }

    /// Create a provider with explicit parameters (useful for tests).
    pub fn new_with_params(
        region: impl Into<String>,
        project_id: impl Into<String>,
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
            region: region.into(),
            project_id: project_id.into(),
            models,
            client,
        }
    }

    /// Build the Vertex AI generateContent endpoint URL.
    fn endpoint_url(&self, model: &str, streaming: bool) -> String {
        let action = if streaming {
            "streamGenerateContent?alt=sse"
        } else {
            "generateContent"
        };
        format!(
            "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:{action}",
            region = self.region,
            project = self.project_id,
            model = model,
            action = action,
        )
    }

    /// Build the Gemini `generateContent` request body from a [`ProxiedRequest`].
    fn build_request_body(&self, request: &ProxiedRequest) -> serde_json::Value {
        let system_text = request.system.clone().or_else(|| {
            request
                .messages
                .iter()
                .find(|m| m.role == Role::System)
                .map(|m| m.content.as_text())
        });

        let contents: Vec<serde_json::Value> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|msg| {
                // Vertex uses "user" / "model" roles.
                let role = match msg.role {
                    Role::User | Role::Tool => "user",
                    Role::Assistant => "model",
                    Role::System => unreachable!("system messages filtered"),
                };
                serde_json::json!({
                    "role": role,
                    "parts": [{"text": msg.content.as_text()}],
                })
            })
            .collect();

        let max_output_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
        let mut generation_config = serde_json::json!({
            "maxOutputTokens": max_output_tokens,
        });
        if let Some(temp) = request.temperature {
            generation_config["temperature"] = serde_json::json!(temp);
        }
        if let Some(tp) = request.top_p {
            generation_config["topP"] = serde_json::json!(tp);
        }

        let mut body = serde_json::json!({
            "contents": contents,
            "generationConfig": generation_config,
        });

        if let Some(sys) = system_text {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{"text": sys}]
            });
        }

        body
    }

    /// Parse a Vertex `generateContent` response into a [`ProxiedResponse`].
    fn parse_response(
        body: &serde_json::Value,
        model: &str,
    ) -> Result<ProxiedResponse, SwitchboardError> {
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

        let usage = body.get("usageMetadata").map(|u| Usage {
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

        Ok(ProxiedResponse {
            model: model.to_string(),
            content,
            tool_calls: None,
            finish_reason,
            usage,
        })
    }
}

#[async_trait]
impl UpstreamProvider for VertexProvider {
    fn name(&self) -> &str {
        "vertex"
    }

    fn api_format(&self) -> &str {
        "vertex"
    }

    fn supports_model(&self, model: &str) -> bool {
        if self.models.iter().any(|m| m == model) {
            return true;
        }
        model.starts_with("gemini-")
    }

    async fn send(
        &self,
        request: ProxiedRequest,
        key: &PooledKey,
    ) -> Result<ProxiedResponse, SwitchboardError> {
        let url = self.endpoint_url(&request.model, false);
        let body = self.build_request_body(&request);

        tracing::debug!(
            provider = "vertex",
            model = %request.model,
            url = %url,
            "sending non-streaming request to Vertex AI"
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
                provider = "vertex",
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
        let url = self.endpoint_url(&request.model, true);
        let body = self.build_request_body(&request);

        tracing::debug!(
            provider = "vertex",
            model = %request.model,
            url = %url,
            "sending streaming request to Vertex AI"
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

    async fn health_check(&self, _key: &PooledKey) -> bool {
        // Vertex AI has no public health endpoint.
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
            id: "vertex-key".into(),
            credentials: UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer ya29.test-token"),
                expires_at: None,
            },
            weight: 1.0,
            source: KeySource::Static,
            health: KeyHealth::default(),
        }
    }

    fn make_provider() -> VertexProvider {
        VertexProvider::new_with_params(
            "us-central1",
            "my-gcp-project",
            vec!["gemini-1.5-pro".into(), "gemini-1.5-flash".into()],
            Duration::from_secs(30),
        )
    }

    #[test]
    fn test_provider_name_and_format() {
        let p = make_provider();
        assert_eq!(p.name(), "vertex");
        assert_eq!(p.api_format(), "vertex");
    }

    #[test]
    fn test_supports_model_explicit() {
        let p = make_provider();
        assert!(p.supports_model("gemini-1.5-pro"));
        assert!(p.supports_model("gemini-1.5-flash"));
    }

    #[test]
    fn test_supports_model_prefix() {
        let p =
            VertexProvider::new_with_params("us-central1", "proj", vec![], Duration::from_secs(30));
        assert!(p.supports_model("gemini-2.0-flash-exp"));
        assert!(!p.supports_model("gpt-4o"));
        assert!(!p.supports_model("anthropic.claude-3"));
    }

    #[test]
    fn test_endpoint_url_non_streaming() {
        let p = make_provider();
        let url = p.endpoint_url("gemini-1.5-pro", false);
        assert!(url.contains("us-central1-aiplatform.googleapis.com"));
        assert!(url.contains("my-gcp-project"));
        assert!(url.contains("gemini-1.5-pro"));
        assert!(url.ends_with(":generateContent"));
    }

    #[test]
    fn test_endpoint_url_streaming() {
        let p = make_provider();
        let url = p.endpoint_url("gemini-1.5-pro", true);
        assert!(url.contains(":streamGenerateContent?alt=sse"));
    }

    #[test]
    fn test_build_request_body_basic() {
        let p = make_provider();
        let req = ProxiedRequest {
            model: "gemini-1.5-pro".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("Hello!".into()),
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
        let body = p.build_request_body(&req);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Hello!");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 512);
        assert!(
            (body["generationConfig"]["temperature"].as_f64().unwrap() - 0.5).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_build_request_body_with_system() {
        let p = make_provider();
        let req = ProxiedRequest {
            model: "gemini-1.5-pro".into(),
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
            system: Some("Be concise.".into()),
            extra: Default::default(),
        };
        let body = p.build_request_body(&req);
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be concise.");
    }

    #[test]
    fn test_build_request_body_assistant_role_maps_to_model() {
        let p = make_provider();
        let req = ProxiedRequest {
            model: "gemini-1.5-pro".into(),
            messages: vec![
                switchboard_common::types::Message {
                    role: Role::User,
                    content: switchboard_common::types::MessageContent::Text("Hi".into()),
                    tool_call_id: None,
                    tool_calls: None,
                },
                switchboard_common::types::Message {
                    role: Role::Assistant,
                    content: switchboard_common::types::MessageContent::Text("Hello!".into()),
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
        let body = p.build_request_body(&req);
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents[1]["role"], "model");
    }

    #[test]
    fn test_parse_response_basic() {
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
        let resp = VertexProvider::parse_response(&json, "gemini-1.5-pro").unwrap();
        assert_eq!(resp.content, "Hello from Gemini!");
        assert_eq!(resp.model, "gemini-1.5-pro");
        assert_eq!(resp.finish_reason.as_deref(), Some("STOP"));
        let u = resp.usage.unwrap();
        assert_eq!(u.input_tokens, 10);
        assert_eq!(u.output_tokens, 20);
    }

    #[test]
    fn test_parse_response_no_candidates() {
        let json = serde_json::json!({"candidates": []});
        let resp = VertexProvider::parse_response(&json, "gemini-1.5-pro").unwrap();
        assert_eq!(resp.content, "");
        assert!(resp.finish_reason.is_none());
    }

    #[tokio::test]
    async fn test_health_check_always_true() {
        let p = make_provider();
        let key = make_key();
        assert!(p.health_check(&key).await);
    }

    #[tokio::test]
    async fn test_send_non_streaming_mock() {
        use wiremock::matchers::{header, method, path_regex};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // Build a provider that points at the mock server.
        // We create a mock that matches the path pattern.
        Mock::given(method("POST"))
            .and(path_regex(r".*/models/gemini-1.5-pro:generateContent"))
            .and(header("authorization", "Bearer ya29.test-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello from Vertex!"}],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 7
                }
            })))
            .mount(&mock_server)
            .await;

        // Build a testable provider with a custom URL.
        let provider = VertexProviderTestable {
            inner: make_provider(),
            base_url_override: Some(mock_server.uri()),
        };

        let key = make_key();

        let req = ProxiedRequest {
            model: "gemini-1.5-pro".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("Hello!".into()),
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
        assert_eq!(resp.content, "Hello from Vertex!");
        assert_eq!(resp.finish_reason.as_deref(), Some("STOP"));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
    }

    // ── Test helper ─────────────────────────────────────────────────────────

    /// Testable wrapper that overrides the endpoint base URL.
    struct VertexProviderTestable {
        inner: VertexProvider,
        base_url_override: Option<String>,
    }

    impl VertexProviderTestable {
        fn endpoint_url(&self, model: &str, streaming: bool) -> String {
            if let Some(base) = &self.base_url_override {
                let action = if streaming {
                    "streamGenerateContent?alt=sse"
                } else {
                    "generateContent"
                };
                format!(
                    "{base}/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:{action}",
                    base = base,
                    project = self.inner.project_id,
                    region = self.inner.region,
                    model = model,
                    action = action,
                )
            } else {
                self.inner.endpoint_url(model, streaming)
            }
        }

        async fn send(
            &self,
            request: ProxiedRequest,
            key: &PooledKey,
        ) -> Result<ProxiedResponse, SwitchboardError> {
            let url = self.endpoint_url(&request.model, false);
            let body = self.inner.build_request_body(&request);

            let response = self
                .inner
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
                return Err(SwitchboardError::Upstream(format!(
                    "upstream HTTP {status}: {err_body}"
                )));
            }

            let json: serde_json::Value = response.json().await.map_err(|e| {
                SwitchboardError::Upstream(format!("failed to parse response: {e}"))
            })?;

            VertexProvider::parse_response(&json, &request.model)
        }
    }
}
