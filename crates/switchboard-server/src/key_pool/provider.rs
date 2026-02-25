//! Key source variants and the `PooledKey` type.

use crate::auth::UpstreamCredentials;
use crate::key_pool::KeyHealth;

// ── Key source ────────────────────────────────────────────────────────────────

/// Where a pooled key came from — determines how it is refreshed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeySource {
    /// Loaded from the static TOML config.
    Static,
    /// Sourced from HashiCorp Vault.
    Vault { path: String },
    /// Sourced via AWS STS `AssumeRole`.
    AwsSts { role_arn: String },
    /// Added at runtime via the admin API.
    AdminApi,
}

impl std::fmt::Display for KeySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeySource::Static => write!(f, "static"),
            KeySource::Vault { path } => write!(f, "vault:{path}"),
            KeySource::AwsSts { role_arn } => write!(f, "aws_sts:{role_arn}"),
            KeySource::AdminApi => write!(f, "admin_api"),
        }
    }
}

// ── PooledKey ─────────────────────────────────────────────────────────────────

/// A single upstream credential entry managed by the key pool.
#[derive(Debug, Clone)]
pub struct PooledKey {
    /// Unique identifier within this pool (from config `id` field).
    pub id: String,
    /// The HTTP credential to inject into upstream requests.
    pub credentials: UpstreamCredentials,
    /// Configured selection weight (0.0–1.0). Combined with health multiplier.
    pub weight: f64,
    /// How this key was provisioned.
    pub source: KeySource,
    /// Live health metrics updated by the proxy layer.
    pub health: KeyHealth,
}

impl PooledKey {
    /// Create a new pool entry from a static API key string.
    pub fn new_static(
        id: impl Into<String>,
        credentials: UpstreamCredentials,
        weight: f64,
    ) -> Self {
        Self {
            id: id.into(),
            credentials,
            weight,
            source: KeySource::Static,
            health: KeyHealth::default(),
        }
    }

    /// Effective weight for selector algorithms: configured weight × health multiplier.
    pub fn effective_weight(&self) -> f64 {
        self.weight * self.health.weight_multiplier()
    }

    /// Returns `true` if the key can serve requests (not disabled).
    pub fn is_eligible(&self) -> bool {
        self.health.status.is_eligible()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use http::{HeaderName, HeaderValue};

    use super::*;

    fn make_key(id: &str, weight: f64) -> PooledKey {
        PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer sk-test"),
                expires_at: None,
            },
            weight,
        )
    }

    #[test]
    fn test_effective_weight_healthy() {
        let k = make_key("k1", 0.8);
        assert!((k.effective_weight() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_effective_weight_degraded() {
        let mut k = make_key("k1", 1.0);
        k.health.status = crate::key_pool::KeyStatus::Degraded;
        // 1.0 * 0.3 = 0.3
        assert!((k.effective_weight() - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_effective_weight_disabled_is_zero() {
        let mut k = make_key("k1", 1.0);
        k.health.status = crate::key_pool::KeyStatus::Disabled;
        assert!((k.effective_weight() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_eligible() {
        let k = make_key("k1", 1.0);
        assert!(k.is_eligible());
        let mut k2 = make_key("k2", 1.0);
        k2.health.status = crate::key_pool::KeyStatus::Disabled;
        assert!(!k2.is_eligible());
    }

    #[test]
    fn test_key_source_display() {
        assert_eq!(KeySource::Static.to_string(), "static");
        assert_eq!(
            KeySource::Vault {
                path: "/secret/key".into()
            }
            .to_string(),
            "vault:/secret/key"
        );
        assert_eq!(
            KeySource::AwsSts {
                role_arn: "arn:aws:iam::123:role/sb".into()
            }
            .to_string(),
            "aws_sts:arn:aws:iam::123:role/sb"
        );
        assert_eq!(KeySource::AdminApi.to_string(), "admin_api");
    }
}
