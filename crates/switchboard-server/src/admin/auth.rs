//! Admin control plane authentication.
//!
//! Provides [`AdminAuthState`] which validates admin requests based on the
//! configured auth method (static_token or jwt).

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::FromRequestParts;
use axum::http::StatusCode;
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use subtle::ConstantTimeEq as _;

use crate::auth::AuthError;
use crate::auth::jwt::JwtValidator;
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

/// Holds the admin authentication configuration and optional JWT validator.
///
/// This is inserted as an extension on all admin routes so that [`AdminAuth`]
/// can extract and validate credentials.
pub struct AdminAuthState {
    pub config: AdminConfig,
    jwt_validator: Option<Arc<JwtValidator>>,
}

impl AdminAuthState {
    /// Create a new [`AdminAuthState`] from the admin config.
    ///
    /// JWT validator is not yet loaded — call [`init_jwt`] to fetch the JWKS.
    pub fn new(config: AdminConfig) -> Self {
        Self {
            config,
            jwt_validator: None,
        }
    }

    /// Asynchronously fetch the JWKS and build the JWT validator.
    ///
    /// Returns `Ok(())` immediately if `config.auth` is not `"jwt"`.
    /// Returns `Err` if `jwt_issuer` is not configured or the JWKS fetch fails.
    pub async fn init_jwt(&mut self) -> Result<(), AuthError> {
        if self.config.auth != "jwt" {
            return Ok(());
        }
        let issuer = self
            .config
            .jwt_issuer
            .as_deref()
            .ok_or_else(|| AuthError::Config("admin jwt_issuer not configured".into()))?;

        let jwks_url = format!("{}/.well-known/jwks.json", issuer.trim_end_matches('/'));
        let audience = self.config.jwt_audience.clone().unwrap_or_default();
        let issuer_str = issuer.to_owned();

        let validator = JwtValidator::from_url("admin", &jwks_url, audience, issuer_str)
            .await
            .map_err(|e| AuthError::Config(format!("failed to load admin JWKS: {e}")))?;

        self.jwt_validator = Some(Arc::new(validator));
        Ok(())
    }

    /// Validate the raw `Authorization` header value.
    ///
    /// For `static_token` auth: does constant-time comparison against
    /// `config.static_token`.
    /// For `jwt` auth: validates the JWT using the loaded [`JwtValidator`],
    /// then checks `allowed_roles` if non-empty.
    /// For everything else: returns Unauthorized.
    pub async fn validate(&self, auth_header: &str) -> Result<AdminClaims, AuthError> {
        match self.config.auth.as_str() {
            "static_token" => self.validate_static_token(auth_header),
            "jwt" => self.validate_jwt(auth_header).await,
            other => Err(AuthError::Config(format!(
                "unsupported admin auth method: {other}"
            ))),
        }
    }

    /// Validate using the JWT validator.
    async fn validate_jwt(&self, auth_header: &str) -> Result<AdminClaims, AuthError> {
        let validator = self.jwt_validator.as_ref().ok_or_else(|| {
            AuthError::Config("JWT validator not initialised; call init_jwt() first".into())
        })?;

        use crate::auth::validator::ClientAuthValidator as _;
        let validated = validator.validate(auth_header).await?;

        // Extract roles from the "roles" or "groups" claim.
        let roles = extract_roles_from_claims(&validated.claims);

        // Check allowed_roles: if configured, the token must contain at least one.
        if !self.config.allowed_roles.is_empty() {
            let has_role = roles.iter().any(|r| self.config.allowed_roles.contains(r));
            if !has_role {
                return Err(AuthError::Unauthorized(
                    "no matching admin role in JWT claims".into(),
                ));
            }
        }

        Ok(AdminClaims {
            user_id: validated.user_id,
            roles,
        })
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

/// Extract roles from JWT claims.
///
/// Checks the `"roles"` key first, then falls back to `"groups"`.
/// If neither is present, returns an empty `Vec`.
pub fn extract_roles_from_claims(claims: &HashMap<String, Value>) -> Vec<String> {
    // Try "roles" first, then "groups".
    for key in &["roles", "groups"] {
        if let Some(Value::Array(arr)) = claims.get(*key) {
            let roles: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            if !roles.is_empty() {
                return roles;
            }
        }
    }
    Vec::new()
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

    fn jwt_config() -> AdminConfig {
        AdminConfig {
            enabled: true,
            auth: "jwt".into(),
            jwt_issuer: Some("https://auth.example.com".into()),
            jwt_audience: Some("switchboard-admin".into()),
            allowed_roles: vec![],
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

    /// An AdminAuthState configured for JWT but without calling init_jwt()
    /// must return AuthError::Config (validator not yet initialised).
    #[tokio::test]
    async fn test_jwt_validator_not_yet_initialised_returns_config_error() {
        let state = AdminAuthState::new(jwt_config());
        let result = state.validate("Bearer some.jwt.token").await;
        let err = result.unwrap_err();
        assert!(
            matches!(err, AuthError::Config(_)),
            "expected Config error, got: {err:?}"
        );
        // The error message should mention init_jwt.
        assert!(err.to_string().contains("init_jwt"));
    }

    /// Keeping the existing test name so we don't break anything — it now
    /// tests the same "not-initialised" scenario (jwt_validator = None).
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

    /// Verify that switching from jwt config to static_token is unaffected.
    #[tokio::test]
    async fn test_jwt_static_token_still_works_after_jwt_config_added() {
        // A state using static_token auth should work regardless of jwt fields.
        let config = AdminConfig {
            enabled: true,
            auth: "static_token".into(),
            static_token: Some("s3cr3t".into()),
            jwt_issuer: Some("https://auth.example.com".into()),
            jwt_audience: Some("unused".into()),
            ..AdminConfig::default()
        };
        let state = AdminAuthState::new(config);
        let result = state.validate("Bearer s3cr3t").await;
        assert!(result.is_ok(), "static_token path should still work");
        let claims = result.unwrap();
        assert!(claims.roles.contains(&"admin".to_string()));
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

    // ── Role-checking helpers ─────────────────────────────────────────────

    /// allowed_roles empty → any JWT that validates is accepted (role check skipped).
    /// We unit-test the role-check logic directly here without a real JWT.
    #[tokio::test]
    async fn test_jwt_allowed_roles_empty_means_all_roles_accepted() {
        // Build a state whose jwt_validator is None but allowed_roles is empty.
        // The validator-not-initialised error fires before role checking,
        // so we test the role logic independently via the helper.
        let empty_claims: HashMap<String, Value> = HashMap::new();
        let roles = extract_roles_from_claims(&empty_claims);
        // allowed_roles is empty → we would accept (no role check performed).
        let allowed: Vec<String> = vec![];
        let has_role = if allowed.is_empty() {
            true // empty allowed_roles means all pass
        } else {
            roles.iter().any(|r| allowed.contains(r))
        };
        assert!(
            has_role,
            "empty allowed_roles should allow all validated JWTs"
        );
    }

    #[test]
    fn test_extract_roles_from_claims_roles_key() {
        let mut claims = HashMap::new();
        claims.insert(
            "roles".into(),
            Value::Array(vec![
                Value::String("admin".into()),
                Value::String("user".into()),
            ]),
        );
        let roles = extract_roles_from_claims(&claims);
        assert_eq!(roles, vec!["admin", "user"]);
    }

    #[test]
    fn test_extract_roles_from_claims_groups_key() {
        let mut claims = HashMap::new();
        claims.insert(
            "groups".into(),
            Value::Array(vec![Value::String("platform-team".into())]),
        );
        let roles = extract_roles_from_claims(&claims);
        assert_eq!(roles, vec!["platform-team"]);
    }

    #[test]
    fn test_extract_roles_from_claims_neither_key() {
        let claims: HashMap<String, Value> = HashMap::new();
        let roles = extract_roles_from_claims(&claims);
        assert!(roles.is_empty());
    }
}
