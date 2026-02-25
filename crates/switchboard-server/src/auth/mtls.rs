//! mTLS client authentication validator (stub).
//!
//! Mutual TLS authentication extracts the client certificate from the TLS
//! connection layer, which is not accessible via the `Authorization` header.
//! Full implementation is deferred to the axum TLS wiring phase.
//!
//! For now, `MtlsValidator` always returns a configuration error to prevent
//! silent misuse.

use async_trait::async_trait;

use crate::auth::validator::{AuthError, ClientAuthValidator, ValidatedClient};

/// Stub mTLS validator.
///
/// mTLS certificate extraction requires access to the raw TLS stream, which
/// is wired up in the axum transport layer phase.  Until that work is complete,
/// this validator always returns [`AuthError::Config`] to make the limitation
/// explicit rather than silently falling through.
#[derive(Debug, Clone)]
pub struct MtlsValidator {
    name: String,
    /// Path to the trusted CA certificate bundle.  Stored for future use.
    _ca: Option<String>,
}

impl MtlsValidator {
    /// Construct a new mTLS validator stub.
    ///
    /// `ca` is the optional path to the CA certificate bundle, stored for
    /// future use when TLS termination is implemented.
    pub fn new(name: impl Into<String>, ca: Option<String>) -> Self {
        Self {
            name: name.into(),
            _ca: ca,
        }
    }
}

#[async_trait]
impl ClientAuthValidator for MtlsValidator {
    fn name(&self) -> &str {
        &self.name
    }

    async fn validate(
        &self,
        _authorization_header_value: &str,
    ) -> Result<ValidatedClient, AuthError> {
        Err(AuthError::Config(
            "mTLS requires TLS termination at the axum layer".into(),
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mtls_validator_returns_config_error() {
        let v = MtlsValidator::new("mtls", None);
        let result = v.validate("Bearer anything").await;
        assert!(matches!(result.unwrap_err(), AuthError::Config(_)));
    }

    #[tokio::test]
    async fn test_mtls_validator_with_ca_returns_config_error() {
        let v = MtlsValidator::new("mtls-prod", Some("/etc/ssl/certs/ca.pem".into()));
        let result = v.validate("").await;
        assert!(matches!(result.unwrap_err(), AuthError::Config(msg) if msg.contains("axum")));
    }

    #[test]
    fn test_mtls_validator_name() {
        let v = MtlsValidator::new("my-mtls", None);
        assert_eq!(v.name(), "my-mtls");
    }

    #[test]
    fn test_mtls_validator_debug() {
        let v = MtlsValidator::new("mtls", Some("ca-path".into()));
        let s = format!("{v:?}");
        assert!(s.contains("MtlsValidator"));
    }
}
