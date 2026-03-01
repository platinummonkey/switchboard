//! Configuration loading and hot-reload for switchboard-server.
//!
//! # Loading order
//! 1. TOML file at the path passed to [`load`].
//! 2. Environment variable overrides with the `SWITCHBOARD` prefix and `__`
//!    as separator (e.g. `SWITCHBOARD__SERVER__LISTEN=0.0.0.0:9000`).
//!
//! # Hot-reload
//! [`HotConfig`] wraps the config in an [`arc_swap::ArcSwap`], providing
//! lock-free reads. Call [`HotConfig::reload`] to atomically swap in a freshly
//! parsed config without restarting.

pub mod admin;
pub mod duration;
pub mod guardrails;
pub mod model_selection;
pub mod observability;
pub mod provider;
pub mod rate_limit;
pub mod routing;

use std::path::Path;
use std::sync::Arc;

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};

pub use admin::AdminConfig;
pub use guardrails::GuardrailsConfig;
pub use model_selection::ModelSelectionConfig;
pub use observability::ObservabilityConfig;
pub use provider::ProvidersConfig;
pub use rate_limit::RateLimitConfig;
pub use routing::RoutingConfig;

use crate::error::ServerError;

// ── Identity ──────────────────────────────────────────────────────────────────

fn default_identity_resolvers() -> Vec<String> {
    vec!["header".into(), "jwt".into(), "api_key".into()]
}

fn default_identity_header() -> String {
    switchboard_common::protocol::HEADER_USER.into()
}

fn default_jwt_claim() -> String {
    "email".into()
}

/// A single API-key to user-identity mapping entry.
///
/// Used by ApiKeyMappingResolver to translate a raw API key value into a
/// known user identity without requiring JWT tokens.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApiKeyMapping {
    /// The plaintext API key value (without the Bearer prefix).
    pub api_key: String,
    /// The user identity to inject when this key matches.
    pub user_id: String,
    /// Optional team identity propagated to the context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub team: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityConfig {
    /// Ordered list of resolver strategies to try.
    #[serde(default = "default_identity_resolvers")]
    pub resolvers: Vec<String>,

    /// Header name for header-based identity.
    #[serde(default = "default_identity_header")]
    pub header_name: String,

    /// JWT claim to extract as the user ID.
    #[serde(default = "default_jwt_claim")]
    pub jwt_claim: String,

    /// Static API key to user identity mappings for ApiKeyMappingResolver.
    #[serde(default)]
    pub api_key_mappings: Vec<ApiKeyMapping>,
}

impl Default for IdentityConfig {
    fn default() -> Self {
        Self {
            resolvers: default_identity_resolvers(),
            header_name: default_identity_header(),
            jwt_claim: default_jwt_claim(),
            api_key_mappings: Vec::new(),
        }
    }
}

// ── Auth validators ───────────────────────────────────────────────────────────

/// A single client→server auth validator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorConfig {
    /// Validator type: "jwt" | "static_keys" | "mtls".
    #[serde(rename = "type")]
    pub validator_type: String,

    // jwt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwks_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audience: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issuer: Option<String>,

    // static_keys
    #[serde(default)]
    pub keys: Vec<String>,

    // mtls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ca: Option<String>,
}

/// All configured client→server auth validators, keyed by a short name.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthConfig {
    #[serde(default)]
    pub validators: std::collections::HashMap<String, ValidatorConfig>,
}

// ── Server listen ─────────────────────────────────────────────────────────────

fn default_listen() -> String {
    "0.0.0.0:8080".into()
}

fn default_shutdown_timeout() -> String {
    "30s".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerListenConfig {
    #[serde(default = "default_listen")]
    pub listen: String,

    #[serde(default = "default_shutdown_timeout")]
    pub graceful_shutdown_timeout: String,
}

impl Default for ServerListenConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            graceful_shutdown_timeout: default_shutdown_timeout(),
        }
    }
}

// ── Root ServerConfig ─────────────────────────────────────────────────────────

/// Complete switchboard-server configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerConfig {
    #[serde(default)]
    pub server: ServerListenConfig,

    #[serde(default)]
    pub admin: AdminConfig,

    #[serde(default)]
    pub identity: IdentityConfig,

    #[serde(default)]
    pub auth: AuthConfig,

    #[serde(default)]
    pub providers: ProvidersConfig,

    #[serde(default)]
    pub model_selection: ModelSelectionConfig,

    #[serde(default)]
    pub routing: RoutingConfig,

    #[serde(default)]
    pub guardrails: GuardrailsConfig,

    #[serde(default)]
    pub observability: ObservabilityConfig,

    #[serde(default)]
    pub rate_limit: RateLimitConfig,
}

// ── Loading ───────────────────────────────────────────────────────────────────

/// Load and deserialize [`ServerConfig`] from a TOML file, with environment
/// variable overrides applied on top.
pub fn load(path: &Path) -> Result<ServerConfig, ServerError> {
    let cfg = config::Config::builder()
        .add_source(config::File::from(path).format(config::FileFormat::Toml))
        .add_source(
            config::Environment::with_prefix("SWITCHBOARD")
                .separator("__")
                .try_parsing(true),
        )
        .build()
        .map_err(|e| ServerError::Config(e.to_string()))?;

    cfg.try_deserialize()
        .map_err(|e| ServerError::Config(e.to_string()))
}

// ── Hot-reload wrapper ────────────────────────────────────────────────────────

/// Thread-safe, lock-free config holder. Reads are always wait-free.
pub struct HotConfig {
    inner: ArcSwap<ServerConfig>,
    path: std::path::PathBuf,
}

impl HotConfig {
    /// Wrap an already-loaded config.
    pub fn new(cfg: ServerConfig, path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            inner: ArcSwap::from_pointee(cfg),
            path: path.into(),
        }
    }

    /// Return a snapshot of the current config. Cheap — no locking.
    pub fn load(&self) -> Arc<ServerConfig> {
        self.inner.load_full()
    }

    /// Re-parse the config file and atomically swap it in.
    pub fn reload(&self) -> Result<(), ServerError> {
        let fresh = load(&self.path)?;
        self.inner.store(Arc::new(fresh));
        Ok(())
    }

    /// Apply a mutation function to the current config and atomically store
    /// the result.
    ///
    /// This clones the current config, applies `f` to the clone, then stores
    /// it — without reading from disk.  Used by the admin API to update
    /// in-memory config sub-sections (model selection, guardrails, routing,
    /// rate limits, key pool entries) without requiring a file on disk.
    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut ServerConfig),
    {
        let current = self.inner.load_full();
        let mut updated = (*current).clone();
        f(&mut updated);
        self.inner.store(Arc::new(updated));
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Mutex;

    use tempfile::NamedTempFile;

    use super::*;

    /// Serialise all tests that mutate environment variables so they don't
    /// interfere with each other under parallel test execution.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn write_toml(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_load_minimal_config() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let f = write_toml(
            r#"
[server]
listen = "0.0.0.0:8080"
graceful_shutdown_timeout = "30s"
"#,
        );
        let cfg = load(f.path()).unwrap();
        assert_eq!(cfg.server.listen, "0.0.0.0:8080");
    }

    #[test]
    fn test_load_full_config() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let f = write_toml(include_str!(
            "../../../../config/switchboard-server.example.toml"
        ));
        let cfg = load(f.path()).unwrap();
        assert_eq!(cfg.server.listen, "0.0.0.0:8080");
        assert!(cfg.admin.enabled);
        assert!(cfg.providers.contains_key("anthropic"));
        assert!(cfg.providers.contains_key("openai"));
        assert!(cfg.providers.contains_key("bedrock"));
        assert!(cfg.providers.contains_key("ollama"));
        assert_eq!(cfg.model_selection.mode, "dynamic");
        assert!(cfg.routing.semantic.enabled);
        assert!(cfg.guardrails.enabled);
        assert!(cfg.observability.enabled);
        assert!(cfg.rate_limit.enabled);
    }

    #[test]
    fn test_load_defaults_when_sections_absent() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let f = write_toml("");
        let cfg = load(f.path()).unwrap();
        assert_eq!(cfg.server.listen, "0.0.0.0:8080");
        assert!(!cfg.admin.enabled);
        assert!(!cfg.guardrails.enabled);
        assert!(!cfg.observability.enabled);
        assert!(!cfg.rate_limit.enabled);
        assert_eq!(cfg.model_selection.mode, "dynamic");
    }

    #[test]
    fn test_env_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let f = write_toml(
            r#"[server]
listen = "0.0.0.0:8080"
"#,
        );
        // Safety: ENV_MUTEX ensures no other config-loading test runs concurrently.
        unsafe {
            std::env::set_var("SWITCHBOARD__SERVER__LISTEN", "0.0.0.0:9999");
        }
        let cfg = load(f.path()).unwrap();
        unsafe {
            std::env::remove_var("SWITCHBOARD__SERVER__LISTEN");
        }
        assert_eq!(cfg.server.listen, "0.0.0.0:9999");
    }

    #[test]
    fn test_identity_defaults() {
        let cfg = ServerConfig::default();
        assert!(cfg.identity.resolvers.contains(&"header".to_string()));
        assert_eq!(cfg.identity.jwt_claim, "email");
    }

    #[test]
    fn test_hot_config_reload() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let content = r#"[server]
listen = "0.0.0.0:8080"
"#;
        let f = write_toml(content);
        let cfg = load(f.path()).unwrap();
        let hot = HotConfig::new(cfg, f.path());
        let snap1 = hot.load();
        assert_eq!(snap1.server.listen, "0.0.0.0:8080");
        hot.reload().unwrap();
        let snap2 = hot.load();
        assert_eq!(snap2.server.listen, "0.0.0.0:8080");
    }

    #[test]
    fn test_load_nonexistent_file_errors() {
        let result = load(Path::new("/nonexistent/path/config.toml"));
        assert!(result.is_err());
    }

    // ── Insta snapshots ───────────────────────────────────────────────────────

    /// Snapshot of the default `ServerConfig` as JSON.
    /// Catches accidental changes to default field values.
    #[test]
    fn test_snapshot_server_config_defaults() {
        let cfg = ServerConfig::default();
        insta::assert_json_snapshot!("server_config_defaults", &cfg);
    }

    /// Snapshot of `ModelSelectionConfig` defaults.
    #[test]
    fn test_snapshot_model_selection_defaults() {
        let cfg = crate::config::model_selection::ModelSelectionConfig::default();
        insta::assert_json_snapshot!("model_selection_defaults", &cfg);
    }

    /// Snapshot of `GuardrailsConfig` defaults.
    #[test]
    fn test_snapshot_guardrails_defaults() {
        let cfg = crate::config::guardrails::GuardrailsConfig::default();
        insta::assert_json_snapshot!("guardrails_defaults", &cfg);
    }

    /// Snapshot of `ObservabilityConfig` defaults.
    #[test]
    fn test_snapshot_observability_defaults() {
        let cfg = crate::config::observability::ObservabilityConfig::default();
        insta::assert_json_snapshot!("observability_defaults", &cfg);
    }

    /// Snapshot of the full example config (structure only — provider key values
    /// are env-var references and will not change).
    #[test]
    fn test_snapshot_full_config_structure() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let f = write_toml(include_str!(
            "../../../../config/switchboard-server.example.toml"
        ));
        let cfg = load(f.path()).unwrap();
        // Snapshot key structural fields rather than the full config to avoid
        // provider key changes breaking the snapshot unnecessarily.
        let structure = serde_json::json!({
            "server_listen": cfg.server.listen,
            "admin_enabled": cfg.admin.enabled,
            "admin_auth": cfg.admin.auth,
            "identity_resolvers": cfg.identity.resolvers,
            "model_selection_mode": cfg.model_selection.mode,
            "model_selection_fallback": cfg.model_selection.fallback,
            "provider_names": {
                "has_anthropic": cfg.providers.contains_key("anthropic"),
                "has_openai": cfg.providers.contains_key("openai"),
                "has_bedrock": cfg.providers.contains_key("bedrock"),
                "has_ollama": cfg.providers.contains_key("ollama"),
            },
            "routing_semantic_enabled": cfg.routing.semantic.enabled,
            "routing_rules_count": cfg.routing.semantic.rules.len(),
            "guardrails_enabled": cfg.guardrails.enabled,
            "guardrails_fail_mode": cfg.guardrails.fail_mode,
            "guardrails_engines_count": cfg.guardrails.engines.len(),
            "observability_enabled": cfg.observability.enabled,
            "rate_limit_enabled": cfg.rate_limit.enabled,
            "rate_limit_default_rpm": cfg.rate_limit.default_rpm,
        });
        insta::assert_json_snapshot!("full_config_structure", &structure);
    }
}
