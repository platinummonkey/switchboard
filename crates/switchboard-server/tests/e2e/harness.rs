//! TestHarness and TestHarnessBuilder.

use std::net::SocketAddr;

use wiremock::MockServer;

use switchboard_server::config::guardrails::{EngineConfig, GuardrailsConfig};
use switchboard_server::config::model_selection::ModelSelectionConfig;
use switchboard_server::config::routing::RoutingConfig;

use crate::client::TestClient;
use crate::config::{BuildOpts, ProviderMocks};

/// Per-provider wiremock mock servers.
pub struct HarnessMocks {
    pub openai: Option<MockServer>,
    pub anthropic: Option<MockServer>,
    pub ollama: Option<MockServer>,
    pub bedrock: Option<MockServer>,
    pub vertex: Option<MockServer>,
}

/// A running E2E test harness.
pub struct TestHarness {
    /// Address of the proxy server.
    #[allow(dead_code)]
    pub addr: SocketAddr,
    /// Address of the admin server (None if admin not enabled).
    pub admin_addr: Option<SocketAddr>,
    /// Mock servers keyed by provider.
    pub mocks: HarnessMocks,
    /// Pre-configured HTTP client for this harness.
    pub client: TestClient,
    /// Sending half of the shutdown channel. Dropping this stops the server.
    pub _shutdown: tokio::sync::oneshot::Sender<()>,
}

/// Builder for [`TestHarness`].
pub struct TestHarnessBuilder {
    pub enable_openai: bool,
    pub enable_anthropic: bool,
    pub enable_ollama: bool,
    pub enable_bedrock: bool,
    pub enable_vertex: bool,
    pub admin_enabled: bool,
    pub guardrails: Option<GuardrailsConfig>,
    /// `(default_rpm, default_tpm)` — enables rate limiting when `Some`.
    pub rate_limit: Option<(u32, u32)>,
    /// Override model-selection policy.
    pub model_selection: Option<ModelSelectionConfig>,
    /// Override semantic routing config.
    pub routing: Option<RoutingConfig>,
    /// Extra API keys for the OpenAI pool `(id, api_key_value)` plus selector.
    /// When set, replaces the single default key.
    pub openai_extra_keys: Option<(Vec<(String, String)>, String)>,
}

impl Default for TestHarnessBuilder {
    fn default() -> Self {
        Self {
            enable_openai: false,
            enable_anthropic: false,
            enable_ollama: false,
            enable_bedrock: false,
            enable_vertex: false,
            admin_enabled: false,
            guardrails: None,
            rate_limit: None,
            model_selection: None,
            routing: None,
            openai_extra_keys: None,
        }
    }
}

impl TestHarnessBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_openai(mut self) -> Self {
        self.enable_openai = true;
        self
    }

    pub fn with_anthropic(mut self) -> Self {
        self.enable_anthropic = true;
        self
    }

    pub fn with_ollama(mut self) -> Self {
        self.enable_ollama = true;
        self
    }

    pub fn with_bedrock(mut self) -> Self {
        self.enable_bedrock = true;
        self
    }

    pub fn with_vertex(mut self) -> Self {
        self.enable_vertex = true;
        self
    }

    pub fn with_admin(mut self) -> Self {
        self.admin_enabled = true;
        self
    }

    /// Configure a custom guardrail pipeline for this harness.
    pub fn with_guardrails(mut self, cfg: GuardrailsConfig) -> Self {
        self.guardrails = Some(cfg);
        self
    }

    /// Convenience: enable a `builtin_keyword` pre-request guardrail that
    /// blocks requests containing any of the given `keywords`.
    pub fn with_keyword_guardrail(self, keywords: Vec<String>) -> Self {
        self.with_guardrails(GuardrailsConfig {
            enabled: true,
            fail_mode: "open".into(),
            timeout: "500ms".into(),
            streaming_mode: "async_audit".into(),
            engines: vec![EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "pre_request".into(),
                action: Some("block".into()),
                keywords,
                rules: vec![],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            }],
        })
    }

    /// Enable rate limiting with the given defaults.
    ///
    /// `rpm` = max requests per minute, `tpm` = max tokens per minute.
    pub fn with_rate_limit(mut self, rpm: u32, tpm: u32) -> Self {
        self.rate_limit = Some((rpm, tpm));
        self
    }

    /// Override the model-selection policy (default: dynamic).
    pub fn with_model_selection(mut self, cfg: ModelSelectionConfig) -> Self {
        self.model_selection = Some(cfg);
        self
    }

    /// Override the semantic routing config.
    pub fn with_routing(mut self, cfg: RoutingConfig) -> Self {
        self.routing = Some(cfg);
        self
    }

    /// Configure the OpenAI pool with multiple named keys and a specific
    /// selector strategy (e.g. `"round_robin"`, `"weighted_random"`).
    ///
    /// `keys` is a list of `(id, api_key_value)` pairs.  The server forwards
    /// each key's value as `Authorization: Bearer <api_key_value>` so tests
    /// can inspect which key was selected by examining upstream request headers.
    pub fn with_openai_keys(
        mut self,
        keys: Vec<(impl Into<String>, impl Into<String>)>,
        selector: impl Into<String>,
    ) -> Self {
        let keys = keys
            .into_iter()
            .map(|(id, k)| (id.into(), k.into()))
            .collect();
        self.openai_extra_keys = Some((keys, selector.into()));
        self
    }

    /// Build and start the test harness.
    pub async fn build(self) -> TestHarness {
        // Start wiremock servers for each requested provider.
        let openai_mock = if self.enable_openai {
            Some(MockServer::start().await)
        } else {
            None
        };
        let anthropic_mock = if self.enable_anthropic {
            Some(MockServer::start().await)
        } else {
            None
        };
        let ollama_mock = if self.enable_ollama {
            Some(MockServer::start().await)
        } else {
            None
        };
        let bedrock_mock = if self.enable_bedrock {
            Some(MockServer::start().await)
        } else {
            None
        };
        let vertex_mock = if self.enable_vertex {
            Some(MockServer::start().await)
        } else {
            None
        };

        // Allocate an admin listen address upfront so the config can reference
        // it by string, then we hand a pre-bound listener to start_test_server.
        let admin_listen = if self.admin_enabled {
            let port = crate::port_allocator::allocate();
            format!("127.0.0.1:{}", port)
        } else {
            "127.0.0.1:9090".into()
        };

        let mocks = ProviderMocks {
            openai: openai_mock,
            anthropic: anthropic_mock,
            ollama: ollama_mock,
            bedrock: bedrock_mock,
            vertex: vertex_mock,
        };

        let config = crate::config::build_test_config(
            &mocks,
            &BuildOpts {
                admin_enabled: self.admin_enabled,
                admin_listen,
                guardrails: self.guardrails,
                rate_limit: self.rate_limit,
                model_selection: self.model_selection,
                routing: self.routing,
                openai_extra_keys: self.openai_extra_keys,
            },
        );

        let (addr, admin_addr, shutdown) =
            crate::server::start_test_server(config, self.admin_enabled).await;

        let client = TestClient::new(addr, admin_addr);

        TestHarness {
            addr,
            admin_addr,
            mocks: HarnessMocks {
                openai: mocks.openai,
                anthropic: mocks.anthropic,
                ollama: mocks.ollama,
                bedrock: mocks.bedrock,
                vertex: mocks.vertex,
            },
            client,
            _shutdown: shutdown,
        }
    }
}
