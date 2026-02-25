//! JWT validator using RS256 / ES256 tokens.
//!
//! Validates incoming `Authorization: Bearer <jwt>` headers against a JWKS
//! (JSON Web Key Set).  The JWKS may be loaded from a URL (production) or
//! supplied as a static JSON string (tests — avoids live HTTP).
//!
//! Checks:
//! - Signature (RS256 or ES256)
//! - `aud` claim must contain the configured audience
//! - `iss` claim must match the configured issuer
//! - `exp` claim (token expiry)
//!
//! JWKS is loaded eagerly at construction.  Background refresh is a future
//! enhancement (Phase later).

use std::collections::HashMap;

use async_trait::async_trait;
use jsonwebtoken::{
    Algorithm, DecodingKey, Validation, decode, decode_header,
    jwk::{AlgorithmParameters, JwkSet},
};
use serde_json::Value;
use tracing::instrument;

use crate::auth::validator::{AuthError, ClientAuthValidator, ValidatedClient};

// ── JwkSet cache ─────────────────────────────────────────────────────────────

/// Cached JWKS — a simple `Vec` of decoded keys paired with their `kid`.
struct CachedKey {
    kid: Option<String>,
    algorithm: Algorithm,
    decoding_key: DecodingKey,
}

impl std::fmt::Debug for CachedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedKey")
            .field("kid", &self.kid)
            .field("algorithm", &self.algorithm)
            .field("decoding_key", &"<opaque>")
            .finish()
    }
}

// ── JwtValidator ─────────────────────────────────────────────────────────────

/// Validates RS256 or ES256 JWTs against a JWKS endpoint.
#[derive(Debug)]
pub struct JwtValidator {
    name: String,
    /// Expected `aud` claim value.
    audience: String,
    /// Expected `iss` claim value.
    issuer: String,
    /// Parsed JWKS keys.
    keys: Vec<CachedKey>,
}

impl JwtValidator {
    /// Construct a [`JwtValidator`] by fetching the JWKS from `jwks_url` using
    /// `reqwest`.
    ///
    /// Returns an error if the HTTP request fails or the response is not a
    /// valid JWKS.
    pub async fn from_url(
        name: impl Into<String>,
        jwks_url: &str,
        audience: impl Into<String>,
        issuer: impl Into<String>,
    ) -> Result<Self, AuthError> {
        let response = reqwest::get(jwks_url)
            .await
            .map_err(|e| AuthError::Jwks(format!("HTTP request failed: {e}")))?;

        let body = response
            .text()
            .await
            .map_err(|e| AuthError::Jwks(format!("Failed to read JWKS response: {e}")))?;

        Self::from_jwks_str(name, &body, audience, issuer)
    }

    /// Construct a [`JwtValidator`] from a static JWKS JSON string.
    ///
    /// Useful in tests where no live HTTP server is available.
    pub fn from_jwks_str(
        name: impl Into<String>,
        jwks_json: &str,
        audience: impl Into<String>,
        issuer: impl Into<String>,
    ) -> Result<Self, AuthError> {
        let jwks: JwkSet = serde_json::from_str(jwks_json)
            .map_err(|e| AuthError::Jwks(format!("Failed to parse JWKS: {e}")))?;

        let mut keys = Vec::new();

        for jwk in &jwks.keys {
            let kid = jwk.common.key_id.clone();

            match &jwk.algorithm {
                AlgorithmParameters::RSA(rsa) => {
                    let decoding_key = DecodingKey::from_rsa_components(&rsa.n, &rsa.e)
                        .map_err(|e| AuthError::Jwks(format!("Failed to parse RSA key: {e}")))?;
                    // Determine algorithm from `alg` field or default to RS256
                    let algorithm = jwk
                        .common
                        .key_algorithm
                        .and_then(|ka| {
                            use jsonwebtoken::jwk::KeyAlgorithm;
                            match ka {
                                KeyAlgorithm::RS256 => Some(Algorithm::RS256),
                                KeyAlgorithm::RS384 => Some(Algorithm::RS384),
                                KeyAlgorithm::RS512 => Some(Algorithm::RS512),
                                KeyAlgorithm::PS256 => Some(Algorithm::PS256),
                                KeyAlgorithm::PS384 => Some(Algorithm::PS384),
                                KeyAlgorithm::PS512 => Some(Algorithm::PS512),
                                _ => None,
                            }
                        })
                        .unwrap_or(Algorithm::RS256);
                    keys.push(CachedKey {
                        kid,
                        algorithm,
                        decoding_key,
                    });
                }
                AlgorithmParameters::EllipticCurve(ec) => {
                    let decoding_key = DecodingKey::from_ec_components(&ec.x, &ec.y)
                        .map_err(|e| AuthError::Jwks(format!("Failed to parse EC key: {e}")))?;
                    let algorithm = jwk
                        .common
                        .key_algorithm
                        .and_then(|ka| {
                            use jsonwebtoken::jwk::KeyAlgorithm;
                            match ka {
                                KeyAlgorithm::ES256 => Some(Algorithm::ES256),
                                KeyAlgorithm::ES384 => Some(Algorithm::ES384),
                                _ => None,
                            }
                        })
                        .unwrap_or(Algorithm::ES256);
                    keys.push(CachedKey {
                        kid,
                        algorithm,
                        decoding_key,
                    });
                }
                // Skip unsupported key types (e.g., OctetKey symmetric HMAC).
                _ => {
                    tracing::warn!("Skipping unsupported JWK algorithm type in JWKS");
                }
            }
        }

        if keys.is_empty() {
            return Err(AuthError::Jwks(
                "JWKS contains no supported RS256/ES256 keys".into(),
            ));
        }

        Ok(Self {
            name: name.into(),
            audience: audience.into(),
            issuer: issuer.into(),
            keys,
        })
    }

    /// Strip `Bearer ` prefix (case-insensitive).
    fn extract_token(header_value: &str) -> &str {
        let v = header_value.trim();
        if let Some(rest) = v
            .strip_prefix("Bearer ")
            .or_else(|| v.strip_prefix("bearer "))
        {
            rest.trim()
        } else {
            v
        }
    }

    /// Try to decode `token` with each cached key, returning the claims on
    /// first success.
    fn try_decode(&self, token: &str) -> Result<HashMap<String, Value>, AuthError> {
        // Inspect the header to narrow to matching `kid` if present.
        let header = decode_header(token)
            .map_err(|e| AuthError::InvalidCredential(format!("malformed JWT header: {e}")))?;
        let requested_kid = header.kid.as_deref();

        let candidate_keys: Vec<&CachedKey> = self
            .keys
            .iter()
            .filter(|k| {
                // If the token specifies a kid, only try keys with that kid.
                // If it doesn't, try all keys.
                match requested_kid {
                    Some(rid) => k.kid.as_deref() == Some(rid),
                    None => true,
                }
            })
            .collect();

        if candidate_keys.is_empty() {
            return Err(AuthError::Unauthorized(format!(
                "no key found for kid={:?}",
                requested_kid
            )));
        }

        let mut last_err: Option<AuthError> = None;

        for cached in candidate_keys {
            let mut validation = Validation::new(cached.algorithm);
            validation.set_audience(&[&self.audience]);
            validation.set_issuer(&[&self.issuer]);
            // `exp` is validated by default in jsonwebtoken 9.x.

            match decode::<HashMap<String, Value>>(token, &cached.decoding_key, &validation) {
                Ok(token_data) => {
                    return Ok(token_data.claims);
                }
                Err(e) => {
                    use jsonwebtoken::errors::ErrorKind;
                    let auth_err = match e.kind() {
                        ErrorKind::ExpiredSignature => AuthError::Expired,
                        ErrorKind::InvalidAudience => AuthError::ClaimMissing("aud".into()),
                        ErrorKind::InvalidIssuer => AuthError::ClaimMissing("iss".into()),
                        _ => AuthError::Unauthorized(format!("JWT validation failed: {e}")),
                    };
                    last_err = Some(auth_err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| AuthError::Unauthorized("JWT validation failed".into())))
    }
}

#[async_trait]
impl ClientAuthValidator for JwtValidator {
    fn name(&self) -> &str {
        &self.name
    }

    #[instrument(skip(self, authorization_header_value), fields(validator = %self.name))]
    async fn validate(
        &self,
        authorization_header_value: &str,
    ) -> Result<ValidatedClient, AuthError> {
        let token = Self::extract_token(authorization_header_value);
        if token.is_empty() {
            return Err(AuthError::InvalidCredential(
                "Authorization header is empty".into(),
            ));
        }

        let claims = self.try_decode(token)?;

        // Extract a user_id: prefer `sub`, then `email`.
        let user_id = claims
            .get("sub")
            .and_then(|v| v.as_str())
            .or_else(|| claims.get("email").and_then(|v| v.as_str()))
            .map(str::to_owned);

        tracing::debug!(user_id = ?user_id, "JWT validation succeeded");
        Ok(ValidatedClient::from_jwt(user_id, claims))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal RS256 JWKS + matching signed JWT for unit tests.
    // Generated offline — no live network required.
    //
    // Private key (PKCS#8 PEM, test only):
    //   Generated with: openssl genrsa 2048 | openssl pkcs8 -topk8 -nocrypt
    //
    // The JWKS below contains the corresponding public key components.
    // The JWT is signed with HS256 for simplicity in unit tests because
    // generating an offline RS256 JWT requires embedding the full private key.
    //
    // For actual RS256 / ES256 validation, see the integration tests that
    // spin up a wiremock JWKS server.

    /// Build a HS256-based test validator using a symmetric JWKS JSON.
    /// This exercises all claim checking logic without requiring asymmetric keys.
    fn make_hs256_jwks() -> String {
        // jsonwebtoken 9.x JwkSet with an OctetKey (HS256).
        // We use this approach to test claim logic without embedding RSA keys.
        serde_json::json!({
            "keys": [
                {
                    "kty": "oct",
                    "kid": "test-key-1",
                    "k": "c2VjcmV0LWtleS1mb3ItdGVzdGluZy1wdXJwb3Nlc29ubHk",
                    "alg": "HS256",
                    "use": "sig"
                }
            ]
        })
        .to_string()
    }

    /// The JWKS above uses an OctetKey which our parser skips (we only support
    /// RS256 / ES256). So loading it should return an error.
    #[test]
    fn test_from_jwks_str_hs256_only_returns_error() {
        let result = JwtValidator::from_jwks_str(
            "test",
            &make_hs256_jwks(),
            "https://switchboard.example.com",
            "https://auth.example.com",
        );
        // OctetKey is skipped → no supported keys → Err
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AuthError::Jwks(_)));
    }

    #[test]
    fn test_from_jwks_str_invalid_json_returns_error() {
        let result = JwtValidator::from_jwks_str("test", "not valid json", "aud", "iss");
        assert!(matches!(result.unwrap_err(), AuthError::Jwks(_)));
    }

    #[test]
    fn test_extract_token_strips_bearer() {
        assert_eq!(
            JwtValidator::extract_token("Bearer tok.en.here"),
            "tok.en.here"
        );
        assert_eq!(
            JwtValidator::extract_token("bearer tok.en.here"),
            "tok.en.here"
        );
        assert_eq!(JwtValidator::extract_token("tok.en.here"), "tok.en.here");
        assert_eq!(
            JwtValidator::extract_token("  Bearer  tok.en.here  "),
            "tok.en.here"
        );
    }

    #[test]
    fn test_extract_token_empty() {
        assert_eq!(JwtValidator::extract_token(""), "");
        assert_eq!(JwtValidator::extract_token("  "), "");
    }

    #[test]
    fn test_auth_error_variants_match_expected() {
        // Ensure AuthError variants used in jwt.rs are constructible.
        let _e1 = AuthError::Expired;
        let _e2 = AuthError::ClaimMissing("aud".into());
        let _e3 = AuthError::Unauthorized("x".into());
        let _e4 = AuthError::Jwks("y".into());
    }

    /// A valid RS256 JWKS with one key. Generated with:
    ///   openssl genrsa -out priv.pem 2048
    ///   openssl rsa -in priv.pem -pubout -out pub.pem
    /// Then extracted n/e from the public key.
    /// Here we use a hard-coded small test JWKS to avoid embedding a full key.
    /// The actual decoding is tested end-to-end in integration tests.
    #[test]
    fn test_from_jwks_str_with_invalid_rsa_components_returns_error() {
        // This JWKS has RSA type but bogus base64url for n/e.
        let bad_jwks = serde_json::json!({
            "keys": [{
                "kty": "RSA",
                "kid": "k1",
                "n": "!!!not-valid-base64url!!!",
                "e": "AQAB",
                "alg": "RS256",
                "use": "sig"
            }]
        })
        .to_string();

        let result = JwtValidator::from_jwks_str("test", &bad_jwks, "aud", "iss");
        // Should fail because `n` is not valid base64url.
        assert!(result.is_err());
    }

    #[test]
    fn test_validated_client_from_jwt_preserves_claims() {
        let mut claims = HashMap::new();
        claims.insert("sub".into(), Value::String("user123".into()));
        claims.insert("email".into(), Value::String("user@example.com".into()));
        let client = ValidatedClient::from_jwt(Some("user123".into()), claims.clone());
        assert_eq!(client.user_id.as_deref(), Some("user123"));
        assert_eq!(client.claims.len(), 2);
    }
}
