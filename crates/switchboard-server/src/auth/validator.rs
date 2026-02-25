//! `ClientAuthValidator` trait — validates incoming client requests.
//!
//! Each validator examines the raw `Authorization` header value and returns
//! either a [`ValidatedClient`] on success or an [`AuthError`] on failure.

use std::collections::HashMap;

use async_trait::async_trait;
use thiserror::Error;

// ── Auth errors (client-side) ─────────────────────────────────────────────────

/// Errors returned by [`ClientAuthValidator`] implementations.
#[derive(Debug, Error)]
pub enum AuthError {
    /// The credential was structurally invalid or could not be parsed.
    #[error("invalid credential: {0}")]
    InvalidCredential(String),

    /// The credential was well-formed but failed validation (wrong key, bad sig, etc.).
    #[error("unauthorized: {0}")]
    Unauthorized(String),

    /// A required JWT claim was missing or had the wrong value.
    #[error("missing or invalid claim: {0}")]
    ClaimMissing(String),

    /// The token has expired.
    #[error("token expired")]
    Expired,

    /// The JWKS could not be loaded or parsed.
    #[error("JWKS error: {0}")]
    Jwks(String),

    /// The validator is misconfigured (startup-time error surfaced at runtime).
    #[error("misconfigured validator: {0}")]
    Config(String),
}

// ── ValidatedClient ───────────────────────────────────────────────────────────

/// The result of successful client authentication.
///
/// Inserted into axum request extensions by the auth middleware so downstream
/// handlers can access the caller's identity without re-validating.
#[derive(Debug, Clone)]
pub struct ValidatedClient {
    /// Resolved user identifier extracted from the credential (e.g. JWT `sub` or
    /// `email` claim).  `None` for static API key auth where no identity is
    /// embedded.
    pub user_id: Option<String>,

    /// All decoded claims (JWT) or metadata (other validators), as JSON values.
    /// Empty for static API key auth.
    pub claims: HashMap<String, serde_json::Value>,
}

impl ValidatedClient {
    /// Convenience constructor for static-key validated clients.
    pub fn from_static_key() -> Self {
        Self {
            user_id: None,
            claims: HashMap::new(),
        }
    }

    /// Convenience constructor for JWT-authenticated clients.
    pub fn from_jwt(user_id: Option<String>, claims: HashMap<String, serde_json::Value>) -> Self {
        Self { user_id, claims }
    }
}

// ── ClientAuthValidator trait ─────────────────────────────────────────────────

/// Validates an incoming client's `Authorization` header and returns a
/// [`ValidatedClient`] on success.
///
/// Implementations are tried in order by [`crate::auth::registry::AuthRegistry`].
/// The first implementation that returns `Ok` wins; `Err` causes the registry
/// to try the next validator.
///
/// Built-in: [`crate::auth::static_key::StaticKeyValidator`],
/// [`crate::auth::jwt::JwtValidator`],
/// [`crate::auth::mtls::MtlsValidator`].
#[async_trait]
pub trait ClientAuthValidator: Send + Sync + 'static {
    /// Short identifier for logging and metrics, e.g. `"static_keys"`.
    fn name(&self) -> &str;

    /// Validate `authorization_header_value` (the raw value of the
    /// `Authorization` HTTP header, including any `Bearer ` prefix).
    ///
    /// Return `Ok(ValidatedClient)` on success, or `Err(AuthError)` if this
    /// validator cannot authenticate the request.
    async fn validate(
        &self,
        authorization_header_value: &str,
    ) -> Result<ValidatedClient, AuthError>;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `ClientAuthValidator` is object-safe.
    fn _assert_object_safe(_: &dyn ClientAuthValidator) {}

    struct AlwaysPass;

    #[async_trait]
    impl ClientAuthValidator for AlwaysPass {
        fn name(&self) -> &str {
            "always_pass"
        }

        async fn validate(&self, _header: &str) -> Result<ValidatedClient, AuthError> {
            Ok(ValidatedClient::from_static_key())
        }
    }

    struct AlwaysFail;

    #[async_trait]
    impl ClientAuthValidator for AlwaysFail {
        fn name(&self) -> &str {
            "always_fail"
        }

        async fn validate(&self, _header: &str) -> Result<ValidatedClient, AuthError> {
            Err(AuthError::Unauthorized("always fails".into()))
        }
    }

    #[tokio::test]
    async fn test_always_pass_returns_ok() {
        let v = AlwaysPass;
        let result = v.validate("Bearer sk-test").await;
        assert!(result.is_ok());
        let client = result.unwrap();
        assert!(client.user_id.is_none());
        assert!(client.claims.is_empty());
    }

    #[tokio::test]
    async fn test_always_fail_returns_err() {
        let v = AlwaysFail;
        let result = v.validate("Bearer sk-test").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_validated_client_from_static_key() {
        let c = ValidatedClient::from_static_key();
        assert!(c.user_id.is_none());
        assert!(c.claims.is_empty());
    }

    #[test]
    fn test_validated_client_from_jwt() {
        let mut claims = HashMap::new();
        claims.insert(
            "email".into(),
            serde_json::Value::String("alice@example.com".into()),
        );
        let c = ValidatedClient::from_jwt(Some("alice@example.com".into()), claims.clone());
        assert_eq!(c.user_id.as_deref(), Some("alice@example.com"));
        assert_eq!(c.claims.len(), 1);
    }

    #[test]
    fn test_auth_error_display() {
        assert!(
            AuthError::InvalidCredential("bad".into())
                .to_string()
                .contains("bad")
        );
        assert!(
            AuthError::Unauthorized("denied".into())
                .to_string()
                .contains("denied")
        );
        assert!(
            AuthError::ClaimMissing("sub".into())
                .to_string()
                .contains("sub")
        );
        assert!(AuthError::Expired.to_string().contains("expired"));
        assert!(
            AuthError::Jwks("timeout".into())
                .to_string()
                .contains("timeout")
        );
        assert!(
            AuthError::Config("missing".into())
                .to_string()
                .contains("missing")
        );
    }

    #[test]
    fn test_validator_name_always_pass() {
        assert_eq!(AlwaysPass.name(), "always_pass");
    }
}
