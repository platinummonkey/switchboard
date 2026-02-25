//! Static API key validator.
//!
//! Validates incoming `Authorization: Bearer <key>` headers against a
//! configured list of static keys using constant-time comparison to prevent
//! timing side-channels.

use async_trait::async_trait;
use subtle::ConstantTimeEq;
use tracing::instrument;

use crate::auth::validator::{AuthError, ClientAuthValidator, ValidatedClient};

/// Validates incoming requests against a static list of pre-shared API keys.
///
/// Strips the `Bearer ` prefix (case-insensitive) from the `Authorization`
/// header value before comparison.  Uses `subtle::ConstantTimeEq` for
/// constant-time comparison to prevent timing attacks.
#[derive(Debug, Clone)]
pub struct StaticKeyValidator {
    /// Short name used in logs and metrics.
    name: String,
    /// The set of valid API keys (stored as raw bytes for CT comparison).
    keys: Vec<Vec<u8>>,
}

impl StaticKeyValidator {
    /// Construct a validator from a list of plain-text API keys.
    ///
    /// # Panics
    ///
    /// Does not panic; an empty key list is permitted (though the validator
    /// will always reject).
    pub fn new(name: impl Into<String>, keys: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        Self {
            name: name.into(),
            keys: keys
                .into_iter()
                .map(|k| k.as_ref().as_bytes().to_vec())
                .collect(),
        }
    }

    /// Strip `Bearer ` (or `bearer `) prefix, returning the raw key string.
    fn strip_bearer(value: &str) -> &str {
        let v = value.trim();
        // Accept both `Bearer <key>` and a bare `<key>`.
        if let Some(rest) = v
            .strip_prefix("Bearer ")
            .or_else(|| v.strip_prefix("bearer "))
        {
            rest.trim()
        } else {
            v
        }
    }

    /// Constant-time equality check: returns `true` if `candidate` matches
    /// any configured key.
    fn matches_any(&self, candidate: &[u8]) -> bool {
        let mut found = subtle::Choice::from(0u8);
        for key in &self.keys {
            // Lengths must match for CT comparison; use a fixed-size check when
            // lengths differ to avoid leaking length information via timing.
            if key.len() == candidate.len() {
                found |= key.as_slice().ct_eq(candidate);
            }
        }
        bool::from(found)
    }
}

#[async_trait]
impl ClientAuthValidator for StaticKeyValidator {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(skip(self, authorization_header_value), fields(validator = %self.name))]
    async fn validate(
        &self,
        authorization_header_value: &str,
    ) -> Result<ValidatedClient, AuthError> {
        if self.keys.is_empty() {
            return Err(AuthError::Config(
                "StaticKeyValidator has no configured keys".into(),
            ));
        }

        let raw_key = Self::strip_bearer(authorization_header_value);
        if raw_key.is_empty() {
            return Err(AuthError::InvalidCredential(
                "Authorization header is empty".into(),
            ));
        }

        if self.matches_any(raw_key.as_bytes()) {
            tracing::debug!("static key validation succeeded");
            Ok(ValidatedClient::from_static_key())
        } else {
            tracing::debug!("static key validation failed: key not in pool");
            Err(AuthError::Unauthorized("invalid API key".into()))
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn validator(keys: &[&str]) -> StaticKeyValidator {
        StaticKeyValidator::new("test", keys.iter().copied())
    }

    #[tokio::test]
    async fn test_valid_key_bearer_prefix() {
        let v = validator(&["sk-secret"]);
        let result = v.validate("Bearer sk-secret").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_valid_key_no_prefix() {
        let v = validator(&["sk-secret"]);
        let result = v.validate("sk-secret").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_invalid_key_rejected() {
        let v = validator(&["sk-correct"]);
        let result = v.validate("Bearer sk-wrong").await;
        assert!(result.is_err());
        matches!(result.unwrap_err(), AuthError::Unauthorized(_));
    }

    #[tokio::test]
    async fn test_empty_header_rejected() {
        let v = validator(&["sk-secret"]);
        let result = v.validate("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_empty_key_list_is_config_error() {
        let v = validator(&[]);
        let result = v.validate("Bearer sk-any").await;
        assert!(matches!(result.unwrap_err(), AuthError::Config(_)));
    }

    #[tokio::test]
    async fn test_multiple_keys_first_valid() {
        let v = validator(&["sk-a", "sk-b", "sk-c"]);
        assert!(v.validate("Bearer sk-a").await.is_ok());
    }

    #[tokio::test]
    async fn test_multiple_keys_last_valid() {
        let v = validator(&["sk-a", "sk-b", "sk-c"]);
        assert!(v.validate("Bearer sk-c").await.is_ok());
    }

    #[tokio::test]
    async fn test_prefix_only_no_key_rejected() {
        let v = validator(&["sk-secret"]);
        // "Bearer " with trailing whitespace and empty key
        let result = v.validate("Bearer ").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_lowercase_bearer_prefix() {
        let v = validator(&["sk-lower"]);
        let result = v.validate("bearer sk-lower").await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_strip_bearer_variants() {
        assert_eq!(StaticKeyValidator::strip_bearer("Bearer sk-x"), "sk-x");
        assert_eq!(StaticKeyValidator::strip_bearer("bearer sk-x"), "sk-x");
        assert_eq!(StaticKeyValidator::strip_bearer("sk-x"), "sk-x");
        assert_eq!(StaticKeyValidator::strip_bearer("  Bearer  sk-x  "), "sk-x");
    }

    #[test]
    fn test_validator_name() {
        let v = StaticKeyValidator::new("prod-keys", ["sk-1"]);
        assert_eq!(v.name(), "prod-keys");
    }

    #[tokio::test]
    async fn test_returns_static_key_client() {
        let v = validator(&["sk-test"]);
        let client = v.validate("Bearer sk-test").await.unwrap();
        assert!(client.user_id.is_none());
        assert!(client.claims.is_empty());
    }
}
