//! Configuration for switchboard-local (`~/.switchboard/config.toml`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::LocalError;

// ── Server ─────────────────────────────────────────────────────────────────────

fn default_server_url() -> String {
    "https://switchboard.internal:8080".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_server_url")]
    pub url: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            url: default_server_url(),
        }
    }
}

// ── Auth ───────────────────────────────────────────────────────────────────────

fn default_auth_method() -> String {
    "api_key".into()
}

fn default_refresh_interval() -> String {
    "15m".into()
}

/// JWT-specific auth options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JwtAuthConfig {
    /// Existing token value (takes precedence over token_command).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,

    /// Shell command whose stdout is a bearer token.
    /// Example: `"vault read -field=token secret/switchboard"`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_command: Option<String>,

    #[serde(default = "default_refresh_interval")]
    pub refresh_interval: String,
}

/// mTLS-specific auth options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MtlsAuthConfig {
    pub cert: String,
    pub key: String,
    pub ca: String,
}

/// OAuth2-specific auth options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OAuthConfig {
    pub client_id: String,
    pub token_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Auth method: "api_key" | "jwt" | "mtls" | "oauth".
    #[serde(default = "default_auth_method")]
    pub method: String,

    /// Static API key (method = "api_key").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    #[serde(default)]
    pub jwt: JwtAuthConfig,

    #[serde(default)]
    pub mtls: MtlsAuthConfig,

    #[serde(default)]
    pub oauth: OAuthConfig,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            method: default_auth_method(),
            api_key: None,
            jwt: JwtAuthConfig::default(),
            mtls: MtlsAuthConfig::default(),
            oauth: OAuthConfig::default(),
        }
    }
}

// ── Identity ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IdentityConfig {
    /// User identifier to inject. Auto-detected from OS username if absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub team: Option<String>,
}

// ── Local listener ─────────────────────────────────────────────────────────────

fn default_listen() -> String {
    "127.0.0.1:8877".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalListenConfig {
    #[serde(default = "default_listen")]
    pub listen: String,
}

impl Default for LocalListenConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
        }
    }
}

// ── Model preferences ─────────────────────────────────────────────────────────

fn default_model() -> String {
    "claude-sonnet-4-20250514".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model to inject when the client does not specify one.
    #[serde(default = "default_model")]
    pub default: String,

    /// Model name overrides: when the tool requests key, send value instead.
    #[serde(default)]
    pub overrides: HashMap<String, String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default: default_model(),
            overrides: HashMap::new(),
        }
    }
}

// ── Root LocalConfig ───────────────────────────────────────────────────────────

/// Complete switchboard-local configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LocalConfig {
    #[serde(default)]
    pub server: ServerConfig,

    #[serde(default)]
    pub auth: AuthConfig,

    #[serde(default)]
    pub identity: IdentityConfig,

    #[serde(default)]
    pub local: LocalListenConfig,

    #[serde(default)]
    pub model: ModelConfig,
}

// ── Loading ────────────────────────────────────────────────────────────────────

/// Default config file path: `~/.switchboard/config.toml`.
pub fn default_config_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".switchboard").join("config.toml")
}

/// Load [`LocalConfig`] from a TOML file. Missing optional sections fall back
/// to defaults.
pub fn load(path: &Path) -> Result<LocalConfig, LocalError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| LocalError::Config(format!("cannot read {}: {e}", path.display())))?;
    toml::from_str(&content)
        .map_err(|e| LocalError::Config(format!("invalid config at {}: {e}", path.display())))
}

/// Load from the default path, falling back to an all-defaults config if the
/// file does not exist yet.
pub fn load_or_default() -> Result<LocalConfig, LocalError> {
    let path = default_config_path();
    if path.exists() {
        load(&path)
    } else {
        Ok(LocalConfig::default())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    fn write_toml(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_load_minimal() {
        let f = write_toml(include_str!("../../../config/switchboard-local.toml"));
        let cfg = load(f.path()).unwrap();
        assert_eq!(cfg.server.url, "https://switchboard.internal:8080");
        assert_eq!(cfg.auth.method, "api_key");
        assert_eq!(cfg.local.listen, "127.0.0.1:8877");
        assert_eq!(cfg.model.default, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_load_example_config() {
        let f = write_toml(include_str!(
            "../../../config/switchboard-local.example.toml"
        ));
        let cfg = load(f.path()).unwrap();
        assert_eq!(cfg.auth.method, "jwt");
        assert_eq!(
            cfg.auth.jwt.token_command.as_deref(),
            Some("vault read -field=token secret/switchboard")
        );
        assert_eq!(
            cfg.model.overrides.get("gpt-4").map(String::as_str),
            Some("claude-sonnet-4-20250514")
        );
    }

    #[test]
    fn test_load_defaults_when_absent() {
        let f = write_toml("");
        let cfg = load(f.path()).unwrap();
        assert_eq!(cfg.server.url, "https://switchboard.internal:8080");
        assert_eq!(cfg.auth.method, "api_key");
        assert_eq!(cfg.model.default, "claude-sonnet-4-20250514");
        assert!(cfg.model.overrides.is_empty());
    }

    #[test]
    fn test_defaults() {
        let cfg = LocalConfig::default();
        assert_eq!(cfg.server.url, "https://switchboard.internal:8080");
        assert_eq!(cfg.local.listen, "127.0.0.1:8877");
        assert_eq!(cfg.model.default, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_model_overrides() {
        let f = write_toml(
            r#"
[model]
default = "claude-sonnet-4-20250514"

[model.overrides]
"gpt-4" = "claude-sonnet-4-20250514"
"gpt-4o" = "claude-sonnet-4-20250514"
"#,
        );
        let cfg = load(f.path()).unwrap();
        assert_eq!(cfg.model.overrides.len(), 2);
        assert_eq!(cfg.model.overrides["gpt-4"], "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_load_nonexistent_errors() {
        assert!(load(Path::new("/no/such/config.toml")).is_err());
    }
}
