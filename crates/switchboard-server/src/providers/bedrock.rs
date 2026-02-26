//! Amazon Bedrock Converse API upstream provider.
//!
//! Uses AWS SigV4 signing for every request. Credentials are conveyed via
//! the [`PooledKey`] using a special convention:
//!
//! - If `header_name == "x-switchboard-bedrock-creds"`, `header_value` is a
//!   JSON string `{"access_key":"…","secret_key":"…","session_token":"…","region":"…"}`.
//! - Otherwise the `header_value` is treated as a static access-key-only stub
//!   (no signing is attempted — used for testing only).
//!
//! # Endpoints
//!
//! Non-streaming: `POST https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse`
//! Streaming:     `POST https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse-stream`
//!
//! Cross-region inference profiles prepend `{profile_prefix}.` to the model
//! ID when `config.cross_region_inference == true`.

use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign as sigv4_sign};
use aws_sigv4::sign::v4;
use futures_util::StreamExt;
use switchboard_common::errors::SwitchboardError;
use switchboard_common::types::{ProxiedRequest, ProxiedResponse, Role, Usage};

use crate::config::provider::ProviderConfig;
use crate::key_pool::PooledKey;
use crate::routing::{BoxStream, UpstreamProvider};

// ── Constants ─────────────────────────────────────────────────────────────────

const DEFAULT_REGION: &str = "us-east-1";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const BEDROCK_SERVICE: &str = "bedrock";

// ── Credentials helper ────────────────────────────────────────────────────────

/// Parsed AWS credentials extracted from a [`PooledKey`].
#[derive(Debug, Clone)]
struct BedrockCreds {
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
    region: String,
}

impl BedrockCreds {
    /// Parse from a `PooledKey`.
    ///
    /// If the header name is `x-switchboard-bedrock-creds`, the value is
    /// decoded as JSON.  Otherwise this returns an error (which causes the
    /// caller to skip signing and pass headers directly — only valid in test
    /// harnesses that bypass SigV4).
    fn from_key(key: &PooledKey, default_region: &str) -> Result<Self, String> {
        let name = key.credentials.header_name.as_str();
        if name == "x-switchboard-bedrock-creds" {
            let val = key
                .credentials
                .header_value
                .to_str()
                .map_err(|e| format!("invalid header value: {e}"))?;
            let parsed: serde_json::Value = serde_json::from_str(val)
                .map_err(|e| format!("failed to parse bedrock creds JSON: {e}"))?;
            let access_key = parsed["access_key"]
                .as_str()
                .ok_or("missing access_key")?
                .to_string();
            let secret_key = parsed["secret_key"]
                .as_str()
                .ok_or("missing secret_key")?
                .to_string();
            let session_token = parsed["session_token"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            let region = parsed["region"]
                .as_str()
                .filter(|s| !s.is_empty())
                .unwrap_or(default_region)
                .to_string();
            Ok(Self {
                access_key,
                secret_key,
                session_token,
                region,
            })
        } else {
            Err(format!(
                "unexpected credential header '{}'; expected 'x-switchboard-bedrock-creds'",
                name
            ))
        }
    }
}

// ── BedrockProvider ───────────────────────────────────────────────────────────

/// Upstream provider for Amazon Bedrock (Converse API + SigV4).
#[derive(Debug)]
pub struct BedrockProvider {
    /// Default AWS region from config.
    region: String,
    /// Whether to use cross-region inference profile URLs.
    cross_region_inference: bool,
    /// Model IDs served by this provider.
    models: Vec<String>,
    /// Shared reqwest client (no default Auth header — we sign each request).
    client: reqwest::Client,
}

impl BedrockProvider {
    /// Create a provider from the server config's provider entry.
    pub fn new(_name: &str, config: &ProviderConfig) -> Result<Self, String> {
        let region = config
            .region
            .clone()
            .unwrap_or_else(|| DEFAULT_REGION.to_string());

        let timeout = parse_duration(&config.timeout).unwrap_or(Duration::from_secs(300));

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| format!("failed to build reqwest client: {e}"))?;

        Ok(Self {
            region,
            cross_region_inference: config.cross_region_inference,
            models: config.models.clone(),
            client,
        })
    }

    /// Create a provider with explicit parameters (useful for tests).
    pub fn new_with_params(
        region: impl Into<String>,
        cross_region_inference: bool,
        models: Vec<String>,
        timeout: Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("failed to build reqwest client");

        Self {
            region: region.into(),
            cross_region_inference,
            models,
            client,
        }
    }

    /// Build the Bedrock Converse API endpoint URL for the given model.
    ///
    /// If cross-region inference is enabled the model ID is prefixed with the
    /// appropriate regional prefix (`"us"` for us-* regions, `"eu"` for eu-*,
    /// `"ap"` for ap-*).
    fn endpoint_url(&self, region: &str, model_id: &str, streaming: bool) -> String {
        let action = if streaming {
            "converse-stream"
        } else {
            "converse"
        };

        let effective_model = if self.cross_region_inference {
            let prefix = region_prefix(region);
            format!("{prefix}.{model_id}")
        } else {
            model_id.to_string()
        };

        format!("https://bedrock-runtime.{region}.amazonaws.com/model/{effective_model}/{action}")
    }

    /// Build a Bedrock Converse request body from a [`ProxiedRequest`].
    fn build_request_body(&self, request: &ProxiedRequest) -> serde_json::Value {
        // Separate system messages.
        let system_text = request.system.clone().or_else(|| {
            request
                .messages
                .iter()
                .find(|m| m.role == Role::System)
                .map(|m| m.content.as_text())
        });

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|msg| {
                let role = match msg.role {
                    Role::User | Role::Tool => "user",
                    Role::Assistant => "assistant",
                    Role::System => unreachable!("system messages filtered"),
                };
                serde_json::json!({
                    "role": role,
                    "content": [{"text": msg.content.as_text()}],
                })
            })
            .collect();

        let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let mut inference_config = serde_json::json!({
            "maxTokens": max_tokens,
        });
        if let Some(temp) = request.temperature {
            inference_config["temperature"] = serde_json::json!(temp);
        }
        if let Some(tp) = request.top_p {
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

    /// Parse a Bedrock Converse API response into a [`ProxiedResponse`].
    fn parse_response(
        body: &serde_json::Value,
        model: &str,
    ) -> Result<ProxiedResponse, SwitchboardError> {
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

        let usage = body.get("usage").map(|u| Usage {
            input_tokens: u.get("inputTokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            output_tokens: u.get("outputTokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
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

    /// Sign a reqwest `RequestBuilder` with AWS SigV4 and send it.
    ///
    /// Returns the signed response or an error.
    async fn sign_and_send(
        &self,
        method: &str,
        url: &str,
        body_bytes: &[u8],
        creds: &BedrockCreds,
    ) -> Result<reqwest::Response, SwitchboardError> {
        let credentials = Credentials::new(
            &creds.access_key,
            &creds.secret_key,
            creds.session_token.clone(),
            None,
            "switchboard",
        );

        let identity = credentials.into();
        let signing_settings = SigningSettings::default();
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(creds.region.as_str())
            .name(BEDROCK_SERVICE)
            .time(SystemTime::now())
            .settings(signing_settings)
            .build()
            .map_err(|e| SwitchboardError::Upstream(format!("SigV4 params error: {e}")))?
            .into();

        let content_type = "application/json";
        let signable_headers = [("content-type", content_type)];

        let signable_request = SignableRequest::new(
            method,
            url,
            signable_headers.iter().copied(),
            SignableBody::Bytes(body_bytes),
        )
        .map_err(|e| SwitchboardError::Upstream(format!("SigV4 signable request error: {e}")))?;

        let (signing_instructions, _signature) = sigv4_sign(signable_request, &signing_params)
            .map_err(|e| SwitchboardError::Upstream(format!("SigV4 signing error: {e}")))?
            .into_parts();

        // Build the reqwest request with the signed headers.
        let mut req_builder = self
            .client
            .post(url)
            .header("content-type", content_type)
            .body(body_bytes.to_vec());

        for (name, value) in signing_instructions.headers() {
            req_builder = req_builder.header(name, value);
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| SwitchboardError::Upstream(format!("request failed: {e}")))?;

        Ok(response)
    }
}

#[async_trait]
impl UpstreamProvider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    fn api_format(&self) -> &str {
        "bedrock"
    }

    fn supports_model(&self, model: &str) -> bool {
        // Check explicit model list first.
        if self.models.iter().any(|m| m == model) {
            return true;
        }
        // Fall back to prefix-based detection.
        model.starts_with("anthropic.")
            || model.starts_with("amazon.")
            || model.starts_with("meta.")
            || model.starts_with("mistral.")
    }

    async fn send(
        &self,
        request: ProxiedRequest,
        key: &PooledKey,
    ) -> Result<ProxiedResponse, SwitchboardError> {
        let creds = BedrockCreds::from_key(key, &self.region)
            .map_err(|e| SwitchboardError::Upstream(format!("credential error: {e}")))?;

        let region = creds.region.clone();
        let url = self.endpoint_url(&region, &request.model, false);
        let body = self.build_request_body(&request);
        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| SwitchboardError::Upstream(format!("serialization error: {e}")))?;

        tracing::debug!(
            provider = "bedrock",
            model = %request.model,
            url = %url,
            "sending non-streaming request to Bedrock"
        );

        let response = self
            .sign_and_send("POST", &url, &body_bytes, &creds)
            .await?;

        let status = response.status();
        if !status.is_success() {
            let err_body = response.text().await.unwrap_or_default();
            tracing::warn!(
                provider = "bedrock",
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
        let creds = BedrockCreds::from_key(key, &self.region)
            .map_err(|e| SwitchboardError::Upstream(format!("credential error: {e}")))?;

        let region = creds.region.clone();
        let url = self.endpoint_url(&region, &request.model, true);
        let body = self.build_request_body(&request);
        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| SwitchboardError::Upstream(format!("serialization error: {e}")))?;

        tracing::debug!(
            provider = "bedrock",
            model = %request.model,
            url = %url,
            "sending streaming request to Bedrock"
        );

        let response = self
            .sign_and_send("POST", &url, &body_bytes, &creds)
            .await?;

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
        // Bedrock has no public health endpoint.
        true
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the cross-region inference profile prefix for a given AWS region.
fn region_prefix(region: &str) -> &'static str {
    if region.starts_with("us-") {
        "us"
    } else if region.starts_with("eu-") {
        "eu"
    } else if region.starts_with("ap-") {
        "ap"
    } else {
        "us"
    }
}

/// Parse a human-readable duration string like "300s", "5m", "1h".
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

    fn make_bedrock_key() -> PooledKey {
        let creds = serde_json::json!({
            "access_key": "AKIAIOSFODNN7EXAMPLE",
            "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "session_token": "",
            "region": "us-east-1"
        });
        PooledKey {
            id: "bedrock-key".into(),
            credentials: UpstreamCredentials {
                header_name: HeaderName::from_static("x-switchboard-bedrock-creds"),
                header_value: HeaderValue::from_str(&creds.to_string()).unwrap(),
                expires_at: None,
            },
            weight: 1.0,
            source: KeySource::Static,
            health: KeyHealth::default(),
        }
    }

    fn make_provider() -> BedrockProvider {
        BedrockProvider::new_with_params("us-east-1", false, vec![], Duration::from_secs(30))
    }

    #[test]
    fn test_provider_name_and_format() {
        let p = make_provider();
        assert_eq!(p.name(), "bedrock");
        assert_eq!(p.api_format(), "bedrock");
    }

    #[test]
    fn test_supports_model_prefixes() {
        let p = make_provider();
        assert!(p.supports_model("anthropic.claude-3-5-sonnet-20241022-v2:0"));
        assert!(p.supports_model("amazon.titan-text-express-v1"));
        assert!(p.supports_model("meta.llama3-8b-instruct-v1:0"));
        assert!(p.supports_model("mistral.mistral-7b-instruct-v0:2"));
        assert!(!p.supports_model("gpt-4o"));
        assert!(!p.supports_model("gemini-1.5-pro"));
    }

    #[test]
    fn test_supports_model_explicit_list() {
        let p = BedrockProvider::new_with_params(
            "us-east-1",
            false,
            vec!["my-custom-model".into()],
            Duration::from_secs(30),
        );
        assert!(p.supports_model("my-custom-model"));
    }

    #[test]
    fn test_endpoint_url_no_cross_region() {
        let p = make_provider();
        let url = p.endpoint_url("us-east-1", "anthropic.claude-3", false);
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3/converse"
        );
    }

    #[test]
    fn test_endpoint_url_streaming() {
        let p = make_provider();
        let url = p.endpoint_url("us-east-1", "anthropic.claude-3", true);
        assert!(url.ends_with("/converse-stream"));
    }

    #[test]
    fn test_endpoint_url_cross_region_us() {
        let p =
            BedrockProvider::new_with_params("us-east-1", true, vec![], Duration::from_secs(30));
        let url = p.endpoint_url("us-east-1", "anthropic.claude-3", false);
        assert!(url.contains("us.anthropic.claude-3"));
    }

    #[test]
    fn test_endpoint_url_cross_region_eu() {
        let p =
            BedrockProvider::new_with_params("eu-west-1", true, vec![], Duration::from_secs(30));
        let url = p.endpoint_url("eu-west-1", "anthropic.claude-3", false);
        assert!(url.contains("eu.anthropic.claude-3"));
    }

    #[test]
    fn test_build_request_body_basic() {
        let p = make_provider();
        let req = ProxiedRequest {
            model: "anthropic.claude-3".into(),
            messages: vec![switchboard_common::types::Message {
                role: Role::User,
                content: switchboard_common::types::MessageContent::Text("Hello!".into()),
                tool_call_id: None,
                tool_calls: None,
            }],
            tools: None,
            stream: false,
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: None,
            system: None,
            extra: Default::default(),
        };
        let body = p.build_request_body(&req);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["text"], "Hello!");
        assert_eq!(body["inferenceConfig"]["maxTokens"], 1024);
        assert!(
            (body["inferenceConfig"]["temperature"].as_f64().unwrap() - 0.7).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_build_request_body_with_system() {
        let p = make_provider();
        let req = ProxiedRequest {
            model: "anthropic.claude-3".into(),
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
            system: Some("You are helpful.".into()),
            extra: Default::default(),
        };
        let body = p.build_request_body(&req);
        assert_eq!(body["system"][0]["text"], "You are helpful.");
        // System must not appear in messages.
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_parse_response_basic() {
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
        let resp = BedrockProvider::parse_response(&json, "anthropic.claude-3").unwrap();
        assert_eq!(resp.content, "Hello from Bedrock!");
        assert_eq!(resp.model, "anthropic.claude-3");
        assert_eq!(resp.finish_reason.as_deref(), Some("end_turn"));
        let u = resp.usage.unwrap();
        assert_eq!(u.input_tokens, 10);
        assert_eq!(u.output_tokens, 5);
    }

    #[test]
    fn test_parse_response_missing_output() {
        let json = serde_json::json!({
            "usage": {"inputTokens": 0, "outputTokens": 0},
            "stopReason": "end_turn"
        });
        let resp = BedrockProvider::parse_response(&json, "anthropic.claude-3").unwrap();
        assert_eq!(resp.content, "");
    }

    #[test]
    fn test_bedrock_creds_from_key() {
        let key = make_bedrock_key();
        let creds = BedrockCreds::from_key(&key, "us-west-2").unwrap();
        assert_eq!(creds.access_key, "AKIAIOSFODNN7EXAMPLE");
        assert_eq!(creds.region, "us-east-1"); // from JSON
        assert!(creds.session_token.is_none()); // empty string → None
    }

    #[test]
    fn test_bedrock_creds_wrong_header_name() {
        let key = PooledKey {
            id: "k".into(),
            credentials: UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer x"),
                expires_at: None,
            },
            weight: 1.0,
            source: KeySource::Static,
            health: KeyHealth::default(),
        };
        assert!(BedrockCreds::from_key(&key, "us-east-1").is_err());
    }

    #[tokio::test]
    async fn test_health_check_always_true() {
        let p = make_provider();
        let key = make_bedrock_key();
        assert!(p.health_check(&key).await);
    }

    #[test]
    fn test_region_prefix() {
        assert_eq!(region_prefix("us-east-1"), "us");
        assert_eq!(region_prefix("us-west-2"), "us");
        assert_eq!(region_prefix("eu-west-1"), "eu");
        assert_eq!(region_prefix("ap-southeast-1"), "ap");
        assert_eq!(region_prefix("unknown"), "us");
    }

    #[tokio::test]
    async fn test_send_non_streaming_mock() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/model/anthropic.claude-3-sonnet/converse"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Hi from Bedrock!"}]
                    }
                },
                "usage": {"inputTokens": 8, "outputTokens": 4},
                "stopReason": "end_turn"
            })))
            .mount(&mock_server)
            .await;

        // Build a provider that points to the mock server by overriding the
        // endpoint URL. We use a custom provider with the mock base URL.
        let provider = BedrockProviderTestable {
            inner: BedrockProvider::new_with_params(
                "us-east-1",
                false,
                vec![],
                Duration::from_secs(10),
            ),
            base_url_override: Some(mock_server.uri()),
        };

        let creds_json = serde_json::json!({
            "access_key": "AKIATEST",
            "secret_key": "testsecret",
            "session_token": "",
            "region": "us-east-1"
        });
        let key = PooledKey {
            id: "k".into(),
            credentials: UpstreamCredentials {
                header_name: HeaderName::from_static("x-switchboard-bedrock-creds"),
                header_value: HeaderValue::from_str(&creds_json.to_string()).unwrap(),
                expires_at: None,
            },
            weight: 1.0,
            source: KeySource::Static,
            health: KeyHealth::default(),
        };

        let req = ProxiedRequest {
            model: "anthropic.claude-3-sonnet".into(),
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
        assert_eq!(resp.content, "Hi from Bedrock!");
        assert_eq!(resp.finish_reason.as_deref(), Some("end_turn"));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 8);
        assert_eq!(usage.output_tokens, 4);
    }

    // ── Test helper ─────────────────────────────────────────────────────────

    /// Testable wrapper that allows overriding the Bedrock endpoint base URL.
    struct BedrockProviderTestable {
        inner: BedrockProvider,
        base_url_override: Option<String>,
    }

    impl BedrockProviderTestable {
        fn endpoint_url(&self, region: &str, model_id: &str, streaming: bool) -> String {
            let action = if streaming {
                "converse-stream"
            } else {
                "converse"
            };
            if let Some(base) = &self.base_url_override {
                let effective_model = if self.inner.cross_region_inference {
                    let prefix = region_prefix(region);
                    format!("{prefix}.{model_id}")
                } else {
                    model_id.to_string()
                };
                format!("{base}/model/{effective_model}/{action}")
            } else {
                self.inner.endpoint_url(region, model_id, streaming)
            }
        }

        async fn send(
            &self,
            request: ProxiedRequest,
            key: &PooledKey,
        ) -> Result<ProxiedResponse, SwitchboardError> {
            let creds = BedrockCreds::from_key(key, &self.inner.region)
                .map_err(|e| SwitchboardError::Upstream(format!("credential error: {e}")))?;
            let region = creds.region.clone();
            let url = self.endpoint_url(&region, &request.model, false);
            let body = self.inner.build_request_body(&request);
            let body_bytes = serde_json::to_vec(&body)
                .map_err(|e| SwitchboardError::Upstream(format!("serialization error: {e}")))?;

            let response = self
                .inner
                .sign_and_send("POST", &url, &body_bytes, &creds)
                .await?;

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

            BedrockProvider::parse_response(&json, &request.model)
        }
    }
}
