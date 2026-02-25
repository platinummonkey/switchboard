//! Rate limiting configuration.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

fn default_rpm() -> u32 {
    60
}

fn default_tpm() -> u32 {
    100_000
}

/// Per-entity rate limit override.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitOverride {
    /// Max requests per minute for this entity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u32>,

    /// Max tokens per minute for this entity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<u32>,
}

/// Global rate limiting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    #[serde(default)]
    pub enabled: bool,

    /// Default max requests per minute per user.
    #[serde(default = "default_rpm")]
    pub default_rpm: u32,

    /// Default max tokens per minute per user.
    #[serde(default = "default_tpm")]
    pub default_tpm: u32,

    /// Per-user or per-team overrides, keyed by user/team identifier.
    #[serde(default)]
    pub overrides: HashMap<String, RateLimitOverride>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_rpm: default_rpm(),
            default_tpm: default_tpm(),
            overrides: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_with_override() {
        let toml = r#"
enabled = true
default_rpm = 60
default_tpm = 100000

[overrides.power-users]
rpm = 120
tpm = 500000
"#;
        let c: RateLimitConfig = toml::from_str(toml).unwrap();
        assert!(c.enabled);
        assert_eq!(c.default_rpm, 60);
        let ov = &c.overrides["power-users"];
        assert_eq!(ov.rpm, Some(120));
        assert_eq!(ov.tpm, Some(500_000));
    }

    #[test]
    fn test_parse_partial_override() {
        let toml = r#"
enabled = true
default_rpm = 30
default_tpm = 50000

[overrides.ci]
tpm = 200000
"#;
        let c: RateLimitConfig = toml::from_str(toml).unwrap();
        let ov = &c.overrides["ci"];
        assert_eq!(ov.rpm, None);
        assert_eq!(ov.tpm, Some(200_000));
    }

    #[test]
    fn test_defaults() {
        let c = RateLimitConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.default_rpm, 60);
        assert_eq!(c.default_tpm, 100_000);
        assert!(c.overrides.is_empty());
    }
}
