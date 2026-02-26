//! Admin control plane authentication.
//!
//! Provides [`AdminAuthState`] which validates admin requests based on the
//! configured auth method (static_token or jwt).

use std::sync::Arc;

use axum::extract::FromRequestParts;
use axum::http::StatusCode;
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq as _;

use crate::auth::AuthError;
use crate::config::AdminConfig;

// ── AdminClaims ───────────────────────────────────────────────────────────────

/// Identity information extracted from a validated admin credential.
#[derive(Debug, Clone)]
pub struct AdminClaims {
    pub user_id: Option<String>,
    pub roles: Vec<String>,
}

// ── AdminAuth extractor ───────────────────────────────────────────────────────

/// Axum extractor that validates admin requests.
///
/// Reads the `Authorization` header and checks it against the [`AdminAuthState`]
/// stored as a request extension.  Returns 401 if the header is missing or
/// invalid.
pub struct AdminAuth(pub AdminClaims);

impl<S> FromRequestParts<S> for AdminAuth
where
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Retrieve the AdminAuthState extension (inserted by serve_admin / router setup).
        let auth_state = parts
            .extensions
            .get::<Arc<AdminAuthState>>()
            .cloned()
            .ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({"error": "admin auth state not configured"})),
                )
                    .into_response()
            })?;

        let header_value = parts
            .headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(serde_json::json!({"error": "missing Authorization header"})),
                )
                    .into_response()
            })?;

        match auth_state.validate(header_value).await {
            Ok(claims) => Ok(AdminAuth(claims)),
            Err(e) => {
                tracing::debug!(error = %e, "admin auth rejected");
                Err((
                    StatusCode::UNAUTHORIZED,
                    axum::Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response())
            }
        }
    }
}

// ── AdminAuthState ────────────────────────────────────────────────────────────

/// Holds the admin authentication configuration and optional auth registry.
///
/// This is inserted as an extension on all admin routes so that [`AdminAuth`]
/// can extract and validate credentials.
pub struct AdminAuthState {
    pub config: AdminConfig,
}

impl AdminAuthState {
    /// Create a new [`AdminAuthState`] from the admin config.
    pub fn new(config: AdminConfig) -> Self {
        Self { config }
    }

    /// Validate the raw `Authorization` header value.
    ///
    /// For `static_token` auth: does constant-time comparison against
    /// `config.static_token`.
    /// For `jwt` auth: currently returns Unauthorized (full JWT impl in Phase 14).
    /// For everything else: returns Unauthorized.
    pub async fn validate(&self, auth_header: &str) -> Result<AdminClaims, AuthError> {
        match self.config.auth.as_str() {
            "static_token" => self.validate_static_token(auth_header),
            "jwt" => {
                // Full JWT validation would use a JwtValidator here.
                // For now, we validate structure but always reject without a
                // real JWKS endpoint configured.
                Err(AuthError::Config(
                    "JWT admin auth requires a JwtValidator; not yet configured".into(),
                ))
            }
            other => Err(AuthError::Config(format!(
                "unsupported admin auth method: {other}"
            ))),
        }
    }

    /// Constant-time comparison against the configured static token.
    fn validate_static_token(&self, auth_header: &str) -> Result<AdminClaims, AuthError> {
        let token = self
            .config
            .static_token
            .as_deref()
            .ok_or_else(|| AuthError::Config("static_token not configured".into()))?;

        if token.is_empty() {
            return Err(AuthError::Config("static_token must not be empty".into()));
        }

        let raw = strip_bearer(auth_header);
        if raw.is_empty() {
            return Err(AuthError::InvalidCredential(
                "Authorization header is empty".into(),
            ));
        }

        // Constant-time comparison: lengths must match, then bytes must match.
        let matched =
            token.len() == raw.len() && bool::from(token.as_bytes().ct_eq(raw.as_bytes()));

        if matched {
            Ok(AdminClaims {
                user_id: None,
                roles: vec!["admin".into()],
            })
        } else {
            Err(AuthError::Unauthorized("invalid admin token".into()))
        }
    }
}

/// Strip the `Bearer ` prefix from an Authorization header value.
fn strip_bearer(value: &str) -> &str {
    let v = value.trim();
    if let Some(rest) = v
        .strip_prefix("Bearer ")
        .or_else(|| v.strip_prefix("bearer "))
    {
        rest.trim()
    } else {
        v
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdminConfig;

    fn static_token_config(token: &str) -> AdminConfig {
        AdminConfig {
            enabled: true,
            auth: "static_token".into(),
            static_token: Some(token.into()),
            ..AdminConfig::default()
        }
    }

    #[tokio::test]
    async fn test_static_token_valid_bearer() {
        let state = AdminAuthState::new(static_token_config("my-secret"));
        let result = state.validate("Bearer my-secret").await;
        assert!(result.is_ok());
        let claims = result.unwrap();
        assert!(claims.roles.contains(&"admin".to_string()));
    }

    #[tokio::test]
    async fn test_static_token_valid_no_prefix() {
        let state = AdminAuthState::new(static_token_config("my-secret"));
        let result = state.validate("my-secret").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_static_token_wrong_token() {
        let state = AdminAuthState::new(static_token_config("correct"));
        let result = state.validate("Bearer wrong").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AuthError::Unauthorized(_)));
    }

    #[tokio::test]
    async fn test_static_token_empty_header() {
        let state = AdminAuthState::new(static_token_config("my-secret"));
        let result = state.validate("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_static_token_not_configured() {
        let config = AdminConfig {
            auth: "static_token".into(),
            static_token: None,
            ..AdminConfig::default()
        };
        let state = AdminAuthState::new(config);
        let result = state.validate("Bearer anything").await;
        assert!(matches!(result.unwrap_err(), AuthError::Config(_)));
    }

    #[tokio::test]
    async fn test_jwt_auth_returns_config_error() {
        let config = AdminConfig {
            auth: "jwt".into(),
            ..AdminConfig::default()
        };
        let state = AdminAuthState::new(config);
        let result = state.validate("Bearer some.jwt.token").await;
        assert!(matches!(result.unwrap_err(), AuthError::Config(_)));
    }

    #[tokio::test]
    async fn test_unknown_auth_method() {
        let config = AdminConfig {
            auth: "mtls".into(),
            ..AdminConfig::default()
        };
        let state = AdminAuthState::new(config);
        let result = state.validate("Bearer x").await;
        assert!(matches!(result.unwrap_err(), AuthError::Config(_)));
    }

    #[test]
    fn test_strip_bearer() {
        assert_eq!(strip_bearer("Bearer sk-x"), "sk-x");
        assert_eq!(strip_bearer("bearer sk-x"), "sk-x");
        assert_eq!(strip_bearer("sk-x"), "sk-x");
        assert_eq!(strip_bearer("  Bearer  sk-x  "), "sk-x");
        assert_eq!(strip_bearer(""), "");
    }
}
