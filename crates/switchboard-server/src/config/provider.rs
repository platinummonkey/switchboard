//! Upstream provider and key pool configuration.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Key entries ────────────────────────────────────────────────────────────────

fn default_key_type() -> String {
    "static".into()
}

fn default_weight() -> f64 {
    1.0
}

/// A single entry in a provider's key pool.
/// The `type` field selects the key source; absent type defaults to "static".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEntry {
    pub id: String,

    /// Source type: "static" | "aws_sts" | "vault". Defaults to "static".
    #[serde(rename = "type", default = "default_key_type")]
    pub key_type: String,

    /// Static API key (for type = "static").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// AWS IAM role ARN (for type = "aws_sts").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role_arn: Option<String>,

    /// AWS region override (for type = "aws_sts").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,

    /// How often to refresh the credential (for type = "aws_sts" or "vault").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_interval: Option<String>,

    /// HashiCorp Vault secret path (for type = "vault").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vault_path: Option<String>,

    /// Selection weight in the pool (0.0–1.0). Default: 1.0.
    #[serde(default = "default_weight")]
    pub weight: f64,
}

// ── Key pool ───────────────────────────────────────────────────────────────────

fn default_selector() -> String {
    "weighted_random".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KeyPoolConfig {
    /// Selection strategy: "weighted_random" | "round_robin" | "least_loaded" | "sticky".
    #[serde(default = "default_selector")]
    pub selector: String,

    #[serde(default)]
    pub keys: Vec<KeyEntry>,
}

// ── Provider ───────────────────────────────────────────────────────────────────

fn default_timeout() -> String {
    "300s".into()
}

fn default_health_check_interval() -> String {
    "30s".into()
}

fn default_max_concurrent() -> u32 {
    100
}

/// Configuration for a single upstream LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Base URL for the API. Not required for Bedrock (uses SDK).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// Wire format: "openai" | "anthropic" | "bedrock" | "vertex".
    pub api_format: String,

    /// Model IDs served by this provider.
    #[serde(default)]
    pub models: Vec<String>,

    /// Default AWS region (Bedrock / Vertex).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,

    /// Enable Bedrock cross-region inference profiles.
    #[serde(default)]
    pub cross_region_inference: bool,

    /// GCP project ID (Vertex).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,

    /// Per-request upstream timeout.
    #[serde(default = "default_timeout")]
    pub timeout: String,

    /// How often to probe the provider's health endpoint.
    #[serde(default = "default_health_check_interval")]
    pub health_check_interval: String,

    /// Maximum simultaneous in-flight requests to this provider.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: u32,

    #[serde(default)]
    pub key_pool: KeyPoolConfig,
}

/// All configured upstream providers, keyed by a short name (e.g. "anthropic").
pub type ProvidersConfig = HashMap<String, ProviderConfig>;

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_provider_toml() -> &'static str {
        r#"
api_format = "anthropic"
models = ["claude-sonnet-4-20250514"]

[key_pool]
[[key_pool.keys]]
id = "k1"
api_key = "sk-test"
"#
    }

    #[test]
    fn test_parse_static_key_defaults() {
        let p: ProviderConfig = toml::from_str(minimal_provider_toml()).unwrap();
        assert_eq!(p.api_format, "anthropic");
        assert_eq!(p.key_pool.keys[0].key_type, "static");
        assert!((p.key_pool.keys[0].weight - 1.0).abs() < f64::EPSILON);
        assert_eq!(p.key_pool.selector, "weighted_random");
        assert_eq!(p.max_concurrent, 100);
    }

    #[test]
    fn test_parse_aws_sts_key() {
        let toml = r#"
api_format = "bedrock"
[key_pool]
selector = "round_robin"
[[key_pool.keys]]
id = "bedrock-east"
type = "aws_sts"
role_arn = "arn:aws:iam::123:role/sb"
region = "us-east-1"
refresh_interval = "45m"
"#;
        let p: ProviderConfig = toml::from_str(toml).unwrap();
        let k = &p.key_pool.keys[0];
        assert_eq!(k.key_type, "aws_sts");
        assert_eq!(k.role_arn.as_deref(), Some("arn:aws:iam::123:role/sb"));
        assert_eq!(k.refresh_interval.as_deref(), Some("45m"));
        assert_eq!(p.key_pool.selector, "round_robin");
    }

    #[test]
    fn test_parse_multiple_keys() {
        let toml = r#"
api_format = "openai"
[key_pool]
[[key_pool.keys]]
id = "k1"
api_key = "sk-one"
weight = 1.0
[[key_pool.keys]]
id = "k2"
api_key = "sk-two"
weight = 0.3
"#;
        let p: ProviderConfig = toml::from_str(toml).unwrap();
        assert_eq!(p.key_pool.keys.len(), 2);
        assert!((p.key_pool.keys[1].weight - 0.3).abs() < f64::EPSILON);
    }
}
