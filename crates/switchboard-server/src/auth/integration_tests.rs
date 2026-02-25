//! Integration tests for auth validators.
//!
//! These tests exercise [`JwtValidator`], [`StaticKeyValidator`], and
//! [`AuthRegistry`] using a real wiremock JWKS server and a real RSA key pair.
//!
//! The RSA test key pair is the well-known one embedded in the `jsonwebtoken`
//! crate's own test suite (n/e components from `rsa_modulus_exponent`).
//!
//! Signing uses `EncodingKey::from_rsa_pem` with a PKCS#8 private key that
//! corresponds to the JWKS public key components below.

#![cfg(test)]

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::{Deserialize, Serialize};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::auth::jwt::JwtValidator;
use crate::auth::registry::AuthRegistry;
use crate::auth::static_key::StaticKeyValidator;
use crate::auth::validator::{AuthError, ClientAuthValidator};

// ── Test RSA key material ─────────────────────────────────────────────────────
//
// Private key: PKCS#8 PEM, 2048-bit RSA.
// Source: jsonwebtoken 9.x test suite (`tests/rsa/private_rsa_key_pkcs8.pem`).
//
// The corresponding public key modulus (n) and exponent (e) are embedded in
// the JWKS constants below so we can build a wiremock JWKS endpoint without
// depending on external files.

const TEST_RSA_PRIVATE_KEY_PKCS8: &str = "-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDJETqse41HRBsc
7cfcq3ak4oZWFCoZlcic525A3FfO4qW9BMtRO/iXiyCCHn8JhiL9y8j5JdVP2Q9Z
IpfElcFd3/guS9w+5RqQGgCR+H56IVUyHZWtTJbKPcwWXQdNUX0rBFcsBzCRESJL
eelOEdHIjG7LRkx5l/FUvlqsyHDVJEQsHwegZ8b8C0fz0EgT2MMEdn10t6Ur1rXz
jMB/wvCg8vG8lvciXmedyo9xJ8oMOh0wUEgxziVDMMovmC+aJctcHUAYubwoGN8T
yzcvnGqL7JSh36Pwy28iPzXZ2RLhAyJFU39vLaHdljwthUaupldlNyCfa6Ofy4qN
ctlUPlN1AgMBAAECggEAdESTQjQ70O8QIp1ZSkCYXeZjuhj081CK7jhhp/4ChK7J
GlFQZMwiBze7d6K84TwAtfQGZhQ7km25E1kOm+3hIDCoKdVSKch/oL54f/BK6sKl
qlIzQEAenho4DuKCm3I4yAw9gEc0DV70DuMTR0LEpYyXcNJY3KNBOTjN5EYQAR9s
2MeurpgK2MdJlIuZaIbzSGd+diiz2E6vkmcufJLtmYUT/k/ddWvEtz+1DnO6bRHh
xuuDMeJA/lGB/EYloSLtdyCF6sII6C6slJJtgfb0bPy7l8VtL5iDyz46IKyzdyzW
tKAn394dm7MYR1RlUBEfqFUyNK7C+pVMVoTwCC2V4QKBgQD64syfiQ2oeUlLYDm4
CcKSP3RnES02bcTyEDFSuGyyS1jldI4A8GXHJ/lG5EYgiYa1RUivge4lJrlNfjyf
dV230xgKms7+JiXqag1FI+3mqjAgg4mYiNjaao8N8O3/PD59wMPeWYImsWXNyeHS
55rUKiHERtCcvdzKl4u35ZtTqQKBgQDNKnX2bVqOJ4WSqCgHRhOm386ugPHfy+8j
m6cicmUR46ND6ggBB03bCnEG9OtGisxTo/TuYVRu3WP4KjoJs2LD5fwdwJqpgtHl
yVsk45Y1Hfo+7M6lAuR8rzCi6kHHNb0HyBmZjysHWZsn79ZM+sQnLpgaYgQGRbKV
DZWlbw7g7QKBgQCl1u+98UGXAP1jFutwbPsx40IVszP4y5ypCe0gqgon3UiY/G+1
zTLp79GGe/SjI2VpQ7AlW7TI2A0bXXvDSDi3/5Dfya9ULnFXv9yfvH1QwWToySpW
Kvd1gYSoiX84/WCtjZOr0e0HmLIb0vw0hqZA4szJSqoxQgvF22EfIWaIaQKBgQCf
34+OmMYw8fEvSCPxDxVvOwW2i7pvV14hFEDYIeZKW2W1HWBhVMzBfFB5SE8yaCQy
pRfOzj9aKOCm2FjjiErVNpkQoi6jGtLvScnhZAt/lr2TXTrl8OwVkPrIaN0bG/AS
aUYxmBPCpXu3UjhfQiWqFq/mFyzlqlgvuCc9g95HPQKBgAscKP8mLxdKwOgX8yFW
GcZ0izY/30012ajdHY+/QK5lsMoxTnn0skdS+spLxaS5ZEO4qvPVb8RAoCkWMMal
2pOhmquJQVDPDLuZHdrIiKiDM20dy9sMfHygWcZjQ4WSxf/J7T9canLZIXFhHAZT
3wc9h4G8BBCtWN2TN/LsGZdB
-----END PRIVATE KEY-----";

/// Base64url-encoded RSA modulus (n) matching the private key above.
/// Taken from the `jsonwebtoken` crate's `rsa_modulus_exponent` test.
const TEST_RSA_N: &str = "yRE6rHuNR0QbHO3H3Kt2pOKGVhQqGZXInOduQNxXzuKlvQTLUTv4l4sggh5_CYYi_cvI-SXVT9kPWSKXxJXBXd_4LkvcPuUakBoAkfh-eiFVMh2VrUyWyj3MFl0HTVF9KwRXLAcwkREiS3npThHRyIxuy0ZMeZfxVL5arMhw1SRELB8HoGfG_AtH89BIE9jDBHZ9dLelK9a184zAf8LwoPLxvJb3Il5nncqPcSfKDDodMFBIMc4lQzDKL5gvmiXLXB1AGLm8KBjfE8s3L5xqi-yUod-j8MtvIj812dkS4QMiRVN_by2h3ZY8LYVGrqZXZTcgn2ujn8uKjXLZVD5TdQ";
const TEST_RSA_E: &str = "AQAB";
const TEST_KID: &str = "test-rsa-key-1";

const TEST_AUDIENCE: &str = "https://switchboard.example.com";
const TEST_ISSUER: &str = "https://auth.example.com";

// ── JWT claim struct ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct TestClaims {
    sub: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    email: Option<String>,
    aud: String,
    iss: String,
    exp: i64,
    iat: i64,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// Build a JWKS JSON string for the test RSA public key.
fn test_jwks() -> String {
    serde_json::json!({
        "keys": [{
            "kty": "RSA",
            "kid": TEST_KID,
            "n": TEST_RSA_N,
            "e": TEST_RSA_E,
            "alg": "RS256",
            "use": "sig"
        }]
    })
    .to_string()
}

/// Sign a JWT with the test RSA private key.
fn sign_jwt(claims: &TestClaims) -> String {
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(TEST_KID.to_string());

    encode(
        &header,
        claims,
        &EncodingKey::from_rsa_pem(TEST_RSA_PRIVATE_KEY_PKCS8.as_bytes()).unwrap(),
    )
    .unwrap()
}

/// Spin up a wiremock server that serves the test JWKS at `/jwks`.
async fn start_jwks_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/jwks"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(test_jwks()),
        )
        .mount(&server)
        .await;
    server
}

/// Build a [`JwtValidator`] pointed at the wiremock JWKS server.
async fn make_jwt_validator(server: &MockServer) -> JwtValidator {
    let jwks_url = format!("{}/jwks", server.uri());
    JwtValidator::from_url("test-jwt", &jwks_url, TEST_AUDIENCE, TEST_ISSUER)
        .await
        .expect("JwtValidator::from_url should succeed against a valid JWKS")
}

// ── JwtValidator tests ────────────────────────────────────────────────────────

/// A valid RS256 token signed with the test key → `Ok(ValidatedClient)`.
#[tokio::test]
async fn test_jwt_validator_valid_token() {
    let server = start_jwks_server().await;
    let validator = make_jwt_validator(&server).await;

    let claims = TestClaims {
        sub: "alice@example.com".into(),
        email: Some("alice@example.com".into()),
        aud: TEST_AUDIENCE.into(),
        iss: TEST_ISSUER.into(),
        exp: now_secs() + 3600,
        iat: now_secs(),
    };
    let token = sign_jwt(&claims);

    let result = validator
        .validate(&format!("Bearer {token}"))
        .await
        .expect("valid token should pass");

    assert_eq!(
        result.user_id.as_deref(),
        Some("alice@example.com"),
        "user_id should be extracted from sub claim"
    );
    assert!(
        result.claims.contains_key("sub"),
        "claims map should contain sub"
    );
}

/// A token with `exp` in the past → `Err(AuthError::Expired)`.
#[tokio::test]
async fn test_jwt_validator_expired_token() {
    let server = start_jwks_server().await;
    let validator = make_jwt_validator(&server).await;

    let claims = TestClaims {
        sub: "alice@example.com".into(),
        email: None,
        aud: TEST_AUDIENCE.into(),
        iss: TEST_ISSUER.into(),
        // Set exp 1 hour in the past.
        exp: now_secs() - 3600,
        iat: now_secs() - 7200,
    };
    let token = sign_jwt(&claims);

    let err = validator
        .validate(&format!("Bearer {token}"))
        .await
        .expect_err("expired token should fail");

    assert!(
        matches!(err, AuthError::Expired),
        "expected AuthError::Expired, got {err:?}"
    );
}

/// A token with the wrong `aud` claim → `Err(AuthError::ClaimMissing("aud"))`.
#[tokio::test]
async fn test_jwt_validator_wrong_audience() {
    let server = start_jwks_server().await;
    let validator = make_jwt_validator(&server).await;

    let claims = TestClaims {
        sub: "alice@example.com".into(),
        email: None,
        aud: "https://wrong-audience.example.com".into(),
        iss: TEST_ISSUER.into(),
        exp: now_secs() + 3600,
        iat: now_secs(),
    };
    let token = sign_jwt(&claims);

    let err = validator
        .validate(&format!("Bearer {token}"))
        .await
        .expect_err("token with wrong audience should fail");

    assert!(
        matches!(err, AuthError::ClaimMissing(ref c) if c == "aud"),
        "expected AuthError::ClaimMissing(\"aud\"), got {err:?}"
    );
}

/// A token with the wrong `iss` claim → `Err(AuthError::ClaimMissing("iss"))`.
#[tokio::test]
async fn test_jwt_validator_wrong_issuer() {
    let server = start_jwks_server().await;
    let validator = make_jwt_validator(&server).await;

    let claims = TestClaims {
        sub: "alice@example.com".into(),
        email: None,
        aud: TEST_AUDIENCE.into(),
        iss: "https://evil-issuer.example.com".into(),
        exp: now_secs() + 3600,
        iat: now_secs(),
    };
    let token = sign_jwt(&claims);

    let err = validator
        .validate(&format!("Bearer {token}"))
        .await
        .expect_err("token with wrong issuer should fail");

    assert!(
        matches!(err, AuthError::ClaimMissing(ref c) if c == "iss"),
        "expected AuthError::ClaimMissing(\"iss\"), got {err:?}"
    );
}

/// `JwtValidator::from_url` against a live wiremock JWKS endpoint succeeds.
#[tokio::test]
async fn test_jwt_validator_from_url_succeeds() {
    let server = start_jwks_server().await;
    let jwks_url = format!("{}/jwks", server.uri());

    let validator =
        JwtValidator::from_url("live-test", &jwks_url, TEST_AUDIENCE, TEST_ISSUER).await;

    assert!(
        validator.is_ok(),
        "JwtValidator::from_url should succeed: {validator:?}"
    );
}

/// `JwtValidator::from_url` against a non-existent URL → `Err(AuthError::Jwks)`.
#[tokio::test]
async fn test_jwt_validator_from_url_unreachable_host_returns_error() {
    let result = JwtValidator::from_url(
        "unreachable",
        "http://127.0.0.1:1", // port 1 should refuse connection
        TEST_AUDIENCE,
        TEST_ISSUER,
    )
    .await;
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), AuthError::Jwks(_)));
}

// ── StaticKeyValidator tests ──────────────────────────────────────────────────

/// A valid Bearer token → `Ok(ValidatedClient)`.
#[tokio::test]
async fn test_static_key_validator_valid() {
    let validator = StaticKeyValidator::new("test", ["sk-correct-key"]);
    let result = validator.validate("Bearer sk-correct-key").await;
    assert!(result.is_ok(), "valid key should pass: {result:?}");
    let client = result.unwrap();
    assert!(client.user_id.is_none());
    assert!(client.claims.is_empty());
}

/// A wrong key → `Err(AuthError::Unauthorized)`.
#[tokio::test]
async fn test_static_key_validator_invalid() {
    let validator = StaticKeyValidator::new("test", ["sk-correct-key"]);
    let err = validator
        .validate("Bearer sk-wrong-key")
        .await
        .expect_err("wrong key should fail");
    assert!(
        matches!(err, AuthError::Unauthorized(_)),
        "expected Unauthorized, got {err:?}"
    );
}

/// A key that matches one of several configured keys → `Ok`.
#[tokio::test]
async fn test_static_key_validator_multiple_keys_any_valid() {
    let validator = StaticKeyValidator::new("test", ["sk-a", "sk-b", "sk-c"]);
    for key in ["sk-a", "sk-b", "sk-c"] {
        assert!(
            validator.validate(&format!("Bearer {key}")).await.is_ok(),
            "key {key} should be valid"
        );
    }
}

/// Empty `Authorization` header → error.
#[tokio::test]
async fn test_static_key_validator_empty_header_rejected() {
    let validator = StaticKeyValidator::new("test", ["sk-key"]);
    let result = validator.validate("").await;
    assert!(result.is_err());
}

// ── AuthRegistry tests ────────────────────────────────────────────────────────

/// Registry with static key first: key passes → uses first result, never tries JWT.
#[tokio::test]
async fn test_auth_registry_first_validator_wins() {
    let server = start_jwks_server().await;
    let jwt_validator = make_jwt_validator(&server).await;
    let static_validator = StaticKeyValidator::new("static", ["sk-pass"]);

    let registry = AuthRegistry::builder()
        .add(static_validator)
        .add(jwt_validator)
        .build();

    // The static key "sk-pass" should be accepted by the first validator.
    let result = registry.validate("Bearer sk-pass").await;
    assert!(result.is_ok(), "first validator should win: {result:?}");
}

/// Registry: first validator (static key) fails, second (JWT) succeeds.
#[tokio::test]
async fn test_auth_registry_falls_through() {
    let server = start_jwks_server().await;
    let jwt_validator = make_jwt_validator(&server).await;
    // Static validator that only accepts "sk-wrong" — so it won't match our JWT.
    let static_validator = StaticKeyValidator::new("static", ["sk-wrong"]);

    let registry = AuthRegistry::builder()
        .add(static_validator)
        .add(jwt_validator)
        .build();

    // Build a valid JWT — this should not match the static key but should
    // pass the JWT validator.
    let claims = TestClaims {
        sub: "bob@example.com".into(),
        email: None,
        aud: TEST_AUDIENCE.into(),
        iss: TEST_ISSUER.into(),
        exp: now_secs() + 3600,
        iat: now_secs(),
    };
    let token = sign_jwt(&claims);

    let result = registry
        .validate(&format!("Bearer {token}"))
        .await
        .expect("JWT should pass on second validator");

    assert_eq!(result.user_id.as_deref(), Some("bob@example.com"));
}

/// Registry: both validators fail → `Err` from the last one.
#[tokio::test]
async fn test_auth_registry_all_fail_returns_last_error() {
    let v1 = StaticKeyValidator::new("v1", ["sk-a"]);
    let v2 = StaticKeyValidator::new("v2", ["sk-b"]);

    let registry = AuthRegistry::builder().add(v1).add(v2).build();

    let err = registry
        .validate("Bearer sk-neither")
        .await
        .expect_err("both validators should fail");

    // Last error from v2 — Unauthorized.
    assert!(
        matches!(err, AuthError::Unauthorized(_)),
        "expected Unauthorized, got {err:?}"
    );
}

/// JWT validator: token with no `kid` in header works when validator has one key.
#[tokio::test]
async fn test_jwt_validator_no_kid_in_header_falls_back_to_all_keys() {
    let server = start_jwks_server().await;
    let validator = make_jwt_validator(&server).await;

    // Build JWT without a kid header.
    let claims = TestClaims {
        sub: "carol@example.com".into(),
        email: None,
        aud: TEST_AUDIENCE.into(),
        iss: TEST_ISSUER.into(),
        exp: now_secs() + 3600,
        iat: now_secs(),
    };

    // Sign without setting kid in the header.
    let header = Header::new(Algorithm::RS256); // no kid
    let token = encode(
        &header,
        &claims,
        &EncodingKey::from_rsa_pem(TEST_RSA_PRIVATE_KEY_PKCS8.as_bytes()).unwrap(),
    )
    .unwrap();

    // Validator should try all keys when no kid is present.
    let result = validator.validate(&format!("Bearer {token}")).await;
    assert!(
        result.is_ok(),
        "token without kid should still validate: {result:?}"
    );
    assert_eq!(
        result.unwrap().user_id.as_deref(),
        Some("carol@example.com")
    );
}

/// JWT validator: sub claim is preferred over email for user_id.
#[tokio::test]
async fn test_jwt_validator_sub_preferred_over_email_for_user_id() {
    let server = start_jwks_server().await;
    let validator = make_jwt_validator(&server).await;

    let claims = TestClaims {
        sub: "user-id-123".into(),
        email: Some("alice@example.com".into()),
        aud: TEST_AUDIENCE.into(),
        iss: TEST_ISSUER.into(),
        exp: now_secs() + 3600,
        iat: now_secs(),
    };
    let token = sign_jwt(&claims);

    let client = validator
        .validate(&format!("Bearer {token}"))
        .await
        .unwrap();

    // `sub` should take precedence over `email`.
    assert_eq!(
        client.user_id.as_deref(),
        Some("user-id-123"),
        "sub should be preferred over email for user_id"
    );
}

/// JWT validator: claims map includes all expected fields.
#[tokio::test]
async fn test_jwt_validator_claims_map_populated() {
    let server = start_jwks_server().await;
    let validator = make_jwt_validator(&server).await;

    let claims = TestClaims {
        sub: "dave@example.com".into(),
        email: Some("dave@example.com".into()),
        aud: TEST_AUDIENCE.into(),
        iss: TEST_ISSUER.into(),
        exp: now_secs() + 3600,
        iat: now_secs(),
    };
    let token = sign_jwt(&claims);

    let client = validator
        .validate(&format!("Bearer {token}"))
        .await
        .unwrap();

    assert!(client.claims.contains_key("sub"));
    assert!(client.claims.contains_key("aud"));
    assert!(client.claims.contains_key("iss"));
    assert!(client.claims.contains_key("exp"));
}

/// JwtValidator: construct from static JWKS string (no HTTP) → valid.
#[test]
fn test_jwt_validator_from_jwks_str_rsa_key_succeeds() {
    let jwks = test_jwks();
    let result = JwtValidator::from_jwks_str("test", &jwks, TEST_AUDIENCE, TEST_ISSUER);
    assert!(result.is_ok(), "valid RSA JWKS should parse: {result:?}");
}

/// Collecting all claims: extra custom claims are preserved in the map.
#[tokio::test]
async fn test_jwt_validator_extra_claims_preserved() {
    use serde_json::Value;

    let server = start_jwks_server().await;
    let validator = make_jwt_validator(&server).await;

    // Add an extra `role` claim alongside the standard ones.
    #[derive(Serialize)]
    struct ExtendedClaims {
        sub: String,
        aud: String,
        iss: String,
        exp: i64,
        iat: i64,
        role: String,
    }

    let claims = ExtendedClaims {
        sub: "eve@example.com".into(),
        aud: TEST_AUDIENCE.into(),
        iss: TEST_ISSUER.into(),
        exp: now_secs() + 3600,
        iat: now_secs(),
        role: "admin".into(),
    };

    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(TEST_KID.to_string());
    let token = encode(
        &header,
        &claims,
        &EncodingKey::from_rsa_pem(TEST_RSA_PRIVATE_KEY_PKCS8.as_bytes()).unwrap(),
    )
    .unwrap();

    let client = validator
        .validate(&format!("Bearer {token}"))
        .await
        .unwrap();

    assert_eq!(
        client.claims.get("role"),
        Some(&Value::String("admin".into())),
        "custom role claim should be in claims map"
    );
}

// ── AuthRegistry with JWT: first validator wins scenario ─────────────────────

/// Registry with JWT first, static key second: JWT token passes → uses JWT result.
#[tokio::test]
async fn test_auth_registry_jwt_first_wins_over_static() {
    let server = start_jwks_server().await;
    let jwt_validator = make_jwt_validator(&server).await;
    let static_validator = StaticKeyValidator::new("static", ["sk-fallback"]);

    let registry = AuthRegistry::builder()
        .add(jwt_validator)
        .add(static_validator)
        .build();

    let claims = TestClaims {
        sub: "frank@example.com".into(),
        email: None,
        aud: TEST_AUDIENCE.into(),
        iss: TEST_ISSUER.into(),
        exp: now_secs() + 3600,
        iat: now_secs(),
    };
    let token = sign_jwt(&claims);

    let result = registry
        .validate(&format!("Bearer {token}"))
        .await
        .expect("JWT should pass with JWT validator first");

    // Should have user_id from JWT — the static validator would not set user_id.
    assert_eq!(result.user_id.as_deref(), Some("frank@example.com"));
}

/// Registry with static key first, key passes → `user_id` is `None` (static key).
#[tokio::test]
async fn test_auth_registry_static_first_user_id_is_none() {
    let server = start_jwks_server().await;
    let jwt_validator = make_jwt_validator(&server).await;
    let static_validator = StaticKeyValidator::new("static", ["sk-good"]);

    let registry = AuthRegistry::builder()
        .add(static_validator)
        .add(jwt_validator)
        .build();

    let result = registry
        .validate("Bearer sk-good")
        .await
        .expect("static key should pass");

    // Static key auth returns no user_id.
    assert!(
        result.user_id.is_none(),
        "static key auth should not set user_id"
    );
}

// ── HashMap claims helper ─────────────────────────────────────────────────────

/// Verify that `ValidatedClient::from_jwt` stores claims correctly.
#[test]
fn test_validated_client_from_jwt_stores_claims() {
    use crate::auth::validator::ValidatedClient;

    let mut claims = HashMap::new();
    claims.insert("sub".into(), serde_json::Value::String("user123".into()));
    claims.insert(
        "email".into(),
        serde_json::Value::String("user@example.com".into()),
    );

    let client = ValidatedClient::from_jwt(Some("user123".into()), claims);
    assert_eq!(client.user_id.as_deref(), Some("user123"));
    assert_eq!(client.claims.len(), 2);
}
