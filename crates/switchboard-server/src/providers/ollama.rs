//! Ollama upstream provider.
//!
//! Ollama speaks the OpenAI-compatible API on port 11434.  This provider is a
//! thin delegation wrapper around [`OpenAiProvider`], substituting `"ollama"`
//! as the provider name while reusing all OpenAI-wire-format logic.

use std::time::Duration;

use async_trait::async_trait;
use switchboard_common::errors::SwitchboardError;
use switchboard_common::types::{ProxiedRequest, ProxiedResponse};

use crate::config::provider::ProviderConfig;
use crate::key_pool::PooledKey;
use crate::providers::OpenAiProvider;
use crate::routing::{BoxStream, UpstreamProvider};

// ── Default ───────────────────────────────────────────────────────────────────

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

// ── OllamaProvider ────────────────────────────────────────────────────────────

/// Upstream provider for Ollama (OpenAI-compatible local LLM server).
///
/// Delegates all wire-format logic to the inner [`OpenAiProvider`].
#[derive(Debug)]
pub struct OllamaProvider(OpenAiProvider);

impl OllamaProvider {
    /// Create a new provider from the server config's provider entry.
    ///
    /// If `base_url` is not set in the config, defaults to
    /// `"http://localhost:11434"`.
    pub fn new(_name: &str, config: &ProviderConfig) -> Result<Self, String> {
        // Provide a sensible default so Ollama users don't have to specify
        // base_url explicitly.
        let mut patched = config.clone();
        if patched.base_url.is_none() {
            patched.base_url = Some(DEFAULT_BASE_URL.to_string());
        }
        // The provider_name for the inner OpenAiProvider should be "ollama".
        let inner = OpenAiProvider::new("ollama", &patched)?;
        Ok(Self(inner))
    }

    /// Create a provider with explicit parameters (useful for tests).
    pub fn new_with_base_url(
        base_url: impl Into<String>,
        models: Vec<String>,
        timeout: Duration,
    ) -> Self {
        Self(OpenAiProvider::new_named(
            "ollama", base_url, models, timeout,
        ))
    }
}

#[async_trait]
impl UpstreamProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn api_format(&self) -> &str {
        "openai"
    }

    fn supports_model(&self, model: &str) -> bool {
        self.0.supports_model(model)
    }

    async fn send(
        &self,
        request: ProxiedRequest,
        key: &PooledKey,
    ) -> Result<ProxiedResponse, SwitchboardError> {
        self.0.send(request, key).await
    }

    async fn send_streaming(
        &self,
        request: ProxiedRequest,
        key: &PooledKey,
    ) -> Result<BoxStream, SwitchboardError> {
        self.0.send_streaming(request, key).await
    }

    async fn health_check(&self, key: &PooledKey) -> bool {
        self.0.health_check(key).await
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
            id: "ollama-key".into(),
            credentials: UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer ollama"),
                expires_at: None,
            },
            weight: 1.0,
            source: KeySource::Static,
            health: KeyHealth::default(),
        }
    }

    fn make_provider(base_url: &str) -> OllamaProvider {
        OllamaProvider::new_with_base_url(
            base_url,
            vec!["llama3.2".into(), "mistral".into()],
            Duration::from_secs(30),
        )
    }

    #[test]
    fn test_provider_name_and_format() {
        let p = make_provider("http://localhost:11434");
        assert_eq!(p.name(), "ollama");
        assert_eq!(p.api_format(), "openai");
    }

    #[test]
    fn test_supports_model_explicit() {
        let p = make_provider("http://localhost:11434");
        assert!(p.supports_model("llama3.2"));
        assert!(p.supports_model("mistral"));
        assert!(!p.supports_model("gpt-4o"));
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
    async fn test_send_non_streaming_mock() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-ollama",
                "object": "chat.completion",
                "model": "llama3.2",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Ollama!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let key = make_key();

        let req = ProxiedRequest {
            model: "llama3.2".into(),
            messages: vec![switchboard_common::types::Message {
                role: switchboard_common::types::Role::User,
                content: switchboard_common::types::MessageContent::Text("Hello".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(128),
            temperature: None,
            top_p: None,
            system: None,
            extra: Default::default(),
        };

        let resp = provider.send(req, &key).await.unwrap();
        assert_eq!(resp.content, "Hello from Ollama!");
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 6);
        assert_eq!(usage.output_tokens, 4);
    }

    #[tokio::test]
    async fn test_send_uses_openai_compatible_endpoint() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Verify Ollama sends to /v1/chat/completions (the OpenAI-compatible endpoint)
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "mistral",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Mistral here!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })))
            .expect(1) // Must be called exactly once.
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri());
        let key = make_key();

        let req = ProxiedRequest {
            model: "mistral".into(),
            messages: vec![switchboard_common::types::Message {
                role: switchboard_common::types::Role::User,
                content: switchboard_common::types::MessageContent::Text("hi".into()),
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

        let resp = provider.send(req, &key).await.unwrap();
        assert_eq!(resp.content, "Mistral here!");
        // wiremock verifies the mock was called once via `.expect(1)`.
    }
}
