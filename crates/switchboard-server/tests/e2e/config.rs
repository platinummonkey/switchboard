//! Build [`ServerConfig`] for E2E tests.
//!
//! Generates an in-memory config pointing all providers at their wiremock
//! mock server URIs, with a single static auth key for client→server auth.

use std::collections::HashMap;

use wiremock::MockServer;

use switchboard_server::config::admin::AdminConfig;
use switchboard_server::config::guardrails::GuardrailsConfig;
use switchboard_server::config::provider::{KeyEntry, KeyPoolConfig, ProviderConfig};
use switchboard_server::config::rate_limit::RateLimitConfig;
use switchboard_server::config::{AuthConfig, ServerConfig, ValidatorConfig};

/// Handles to all optional per-provider wiremock mock servers.
pub struct ProviderMocks {
    pub openai: Option<MockServer>,
    pub anthropic: Option<MockServer>,
    pub ollama: Option<MockServer>,
    pub bedrock: Option<MockServer>,
    pub vertex: Option<MockServer>,
}

/// Options for building the test config.
pub struct BuildOpts {
    /// Whether the admin server should be enabled.
    pub admin_enabled: bool,
    /// Listen address for the admin server (e.g. `"127.0.0.1:19500"`).
    /// Only used when `admin_enabled` is true.
    pub admin_listen: String,
    /// Optional guardrail pipeline configuration.
    pub guardrails: Option<GuardrailsConfig>,
    /// Optional rate limit `(default_rpm, default_tpm)`.
    pub rate_limit: Option<(u32, u32)>,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            admin_enabled: false,
            admin_listen: "127.0.0.1:19500".into(),
            guardrails: None,
            rate_limit: None,
        }
    }
}

/// Build a [`ServerConfig`] suitable for E2E tests.
///
/// All providers present in `mocks` are registered as providers pointing at
/// the corresponding mock server's URI.  A single `static_keys` auth
/// validator with key `"test-api-key"` is configured so `TestClient` requests
/// are accepted without real credentials.
pub fn build_test_config(mocks: &ProviderMocks, opts: &BuildOpts) -> ServerConfig {
    let mut providers = HashMap::new();

    if let Some(ref mock) = mocks.openai {
        providers.insert(
            "openai".to_string(),
            ProviderConfig {
                base_url: Some(mock.uri()),
                api_format: "openai".into(),
                models: vec!["gpt-4o".into(), "gpt-3.5-turbo".into()],
                region: None,
                cross_region_inference: false,
                project_id: None,
                timeout: "30s".into(),
                health_check_interval: "30s".into(),
                max_concurrent: 10,
                key_pool: KeyPoolConfig {
                    selector: "weighted_random".into(),
                    keys: vec![KeyEntry {
                        id: "openai-test-key".into(),
                        key_type: "static".into(),
                        api_key: Some("sk-test-openai".into()),
                        role_arn: None,
                        region: None,
                        refresh_interval: None,
                        vault_path: None,
                        weight: 1.0,
                    }],
                },
            },
        );
    }

    if let Some(ref mock) = mocks.anthropic {
        providers.insert(
            "anthropic".to_string(),
            ProviderConfig {
                base_url: Some(mock.uri()),
                api_format: "anthropic".into(),
                models: vec![
                    "claude-3-5-sonnet-20241022".into(),
                    "claude-3-haiku-20240307".into(),
                ],
                region: None,
                cross_region_inference: false,
                project_id: None,
                timeout: "30s".into(),
                health_check_interval: "30s".into(),
                max_concurrent: 10,
                key_pool: KeyPoolConfig {
                    selector: "weighted_random".into(),
                    keys: vec![KeyEntry {
                        id: "anthropic-test-key".into(),
                        key_type: "static".into(),
                        api_key: Some("sk-ant-test".into()),
                        role_arn: None,
                        region: None,
                        refresh_interval: None,
                        vault_path: None,
                        weight: 1.0,
                    }],
                },
            },
        );
    }

    if let Some(ref mock) = mocks.ollama {
        providers.insert(
            "ollama".to_string(),
            ProviderConfig {
                base_url: Some(mock.uri()),
                api_format: "ollama".into(),
                models: vec!["llama3.2".into(), "mistral".into()],
                region: None,
                cross_region_inference: false,
                project_id: None,
                timeout: "30s".into(),
                health_check_interval: "30s".into(),
                max_concurrent: 10,
                key_pool: KeyPoolConfig {
                    selector: "weighted_random".into(),
                    keys: vec![KeyEntry {
                        id: "ollama-test-key".into(),
                        key_type: "static".into(),
                        api_key: Some("".into()),
                        role_arn: None,
                        region: None,
                        refresh_interval: None,
                        vault_path: None,
                        weight: 1.0,
                    }],
                },
            },
        );
    }

    if let Some(ref mock) = mocks.bedrock {
        providers.insert(
            "bedrock".to_string(),
            ProviderConfig {
                base_url: Some(mock.uri()),
                api_format: "bedrock".into(),
                models: vec!["anthropic.claude-3-5-sonnet-20241022-v2:0".into()],
                region: Some("us-east-1".into()),
                cross_region_inference: false,
                project_id: None,
                timeout: "30s".into(),
                health_check_interval: "30s".into(),
                max_concurrent: 10,
                key_pool: KeyPoolConfig {
                    selector: "weighted_random".into(),
                    keys: vec![KeyEntry {
                        id: "bedrock-test-key".into(),
                        key_type: "static".into(),
                        api_key: Some(
                            r#"{"access_key":"AKIAIOSFODNN7EXAMPLE","secret_key":"test-secret","session_token":null,"region":"us-east-1"}"#.into(),
                        ),
                        role_arn: None,
                        region: None,
                        refresh_interval: None,
                        vault_path: None,
                        weight: 1.0,
                    }],
                },
            },
        );
    }

    if let Some(ref mock) = mocks.vertex {
        providers.insert(
            "vertex".to_string(),
            ProviderConfig {
                base_url: Some(mock.uri()),
                api_format: "vertex".into(),
                models: vec!["gemini-1.5-pro".into(), "gemini-1.5-flash".into()],
                region: Some("us-central1".into()),
                cross_region_inference: false,
                project_id: Some("test-project".into()),
                timeout: "30s".into(),
                health_check_interval: "30s".into(),
                max_concurrent: 10,
                key_pool: KeyPoolConfig {
                    selector: "weighted_random".into(),
                    keys: vec![KeyEntry {
                        id: "vertex-test-key".into(),
                        key_type: "static".into(),
                        api_key: Some("test-vertex-oauth-token".into()),
                        role_arn: None,
                        region: None,
                        refresh_interval: None,
                        vault_path: None,
                        weight: 1.0,
                    }],
                },
            },
        );
    }

    // Single static-keys auth validator so TestClient's "test-api-key" is accepted.
    let mut validators = HashMap::new();
    validators.insert(
        "test".to_string(),
        ValidatorConfig {
            validator_type: "static_keys".into(),
            jwks_url: None,
            audience: None,
            issuer: None,
            keys: vec!["test-api-key".into()],
            ca: None,
        },
    );

    let admin = if opts.admin_enabled {
        AdminConfig {
            enabled: true,
            listen: opts.admin_listen.clone(),
            auth: "static_token".into(),
            static_token: Some("test-api-key".into()),
            jwt_issuer: None,
            jwt_audience: None,
            allowed_roles: vec![],
            mtls_ca: None,
        }
    } else {
        AdminConfig::default()
    };

    let guardrails = opts.guardrails.clone().unwrap_or_default();

    let rate_limit = if let Some((rpm, tpm)) = opts.rate_limit {
        RateLimitConfig {
            enabled: true,
            default_rpm: rpm,
            default_tpm: tpm,
            overrides: Default::default(),
        }
    } else {
        RateLimitConfig::default()
    };

    ServerConfig {
        providers,
        auth: AuthConfig { validators },
        admin,
        guardrails,
        rate_limit,
        ..ServerConfig::default()
    }
}
