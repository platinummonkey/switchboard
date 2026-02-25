//! Admin control plane configuration.

use serde::{Deserialize, Serialize};

fn default_listen() -> String {
    "127.0.0.1:9090".into()
}

fn default_auth() -> String {
    "static_token".into()
}

/// Admin API and UI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Address to bind the admin listener (separate from proxy port).
    #[serde(default = "default_listen")]
    pub listen: String,

    /// Auth method for admin endpoints: "jwt" | "static_token" | "mtls".
    #[serde(default = "default_auth")]
    pub auth: String,

    /// Static bearer token (auth = "static_token").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_token: Option<String>,

    /// JWT issuer URL (auth = "jwt").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwt_issuer: Option<String>,

    /// JWT audience (auth = "jwt").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwt_audience: Option<String>,

    /// Roles that grant admin access (auth = "jwt").
    #[serde(default)]
    pub allowed_roles: Vec<String>,

    /// Path to CA certificate for mTLS admin (auth = "mtls").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtls_ca: Option<String>,
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            listen: default_listen(),
            auth: default_auth(),
            static_token: None,
            jwt_issuer: None,
            jwt_audience: None,
            allowed_roles: Vec::new(),
            mtls_ca: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_static_token() {
        let toml = r#"
enabled = true
listen = "127.0.0.1:9090"
auth = "static_token"
static_token = "super-secret"
"#;
        let c: AdminConfig = toml::from_str(toml).unwrap();
        assert!(c.enabled);
        assert_eq!(c.auth, "static_token");
        assert_eq!(c.static_token.as_deref(), Some("super-secret"));
    }

    #[test]
    fn test_parse_jwt() {
        let toml = r#"
enabled = true
listen = "127.0.0.1:9090"
auth = "jwt"
jwt_issuer = "https://auth.internal"
jwt_audience = "switchboard-admin"
allowed_roles = ["switchboard-admin", "platform-team"]
"#;
        let c: AdminConfig = toml::from_str(toml).unwrap();
        assert_eq!(c.auth, "jwt");
        assert_eq!(c.jwt_issuer.as_deref(), Some("https://auth.internal"));
        assert_eq!(c.allowed_roles.len(), 2);
    }

    #[test]
    fn test_defaults() {
        let c = AdminConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.listen, "127.0.0.1:9090");
        assert_eq!(c.auth, "static_token");
    }
}
