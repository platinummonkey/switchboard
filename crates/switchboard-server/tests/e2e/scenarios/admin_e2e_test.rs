//! E2E scenario: admin_e2e_test
//!
//! End-to-end tests for the admin REST API running on its dedicated TCP port.
//!
//! The admin server is started by `.with_admin()` on the harness builder.
//! All requests go through `TestClient::admin_get` / `admin_post`, which
//! target `http://<admin_addr>/admin/api/v1/<path>` with the bearer token
//! `"test-api-key"` (matching the `static_token` set in `build_test_config`).
//!
//! Key facts from admin_api_test.rs and the admin handler source:
//! - `GET /health`          → `{ "status": "ok" }`
//! - `GET /providers`       → JSON array; each item has an `"id"` string field
//! - `GET /config`          → JSON object with a `"providers"` key
//! - `POST /config/reload`  → 500 when config file does not exist (expected in tests)
//! - `GET /rate-limits`     → 200 (snapshot of current overrides)

use std::io::Write as _;
use std::time::{SystemTime, UNIX_EPOCH};

use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai::mock_chat_ok;

// ── RSA test key material ─────────────────────────────────────────────────────
//
// Reused from `src/auth/integration_tests.rs` — the same well-known PKCS#8
// private key whose public key components are embedded in the JWKS constants.
//
// Private key: PKCS#8 PEM, 2048-bit RSA.
// Source: jsonwebtoken 9.x test suite (`tests/rsa/private_rsa_key_pkcs8.pem`).

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
const TEST_RSA_N: &str = "yRE6rHuNR0QbHO3H3Kt2pOKGVhQqGZXInOduQNxXzuKlvQTLUTv4l4sggh5_CYYi_cvI-SXVT9kPWSKXxJXBXd_4LkvcPuUakBoAkfh-eiFVMh2VrUyWyj3MFl0HTVF9KwRXLAcwkREiS3npThHRyIxuy0ZMeZfxVL5arMhw1SRELB8HoGfG_AtH89BIE9jDBHZ9dLelK9a184zAf8LwoPLxvJb3Il5nncqPcSfKDDodMFBIMc4lQzDKL5gvmiXLXB1AGLm8KBjfE8s3L5xqi-yUod-j8MtvIj812dkS4QMiRVN_by2h3ZY8LYVGrqZXZTcgn2ujn8uKjXLZVD5TdQ";
const TEST_RSA_E: &str = "AQAB";
const TEST_KID: &str = "test-rsa-key-1";

const ADMIN_JWT_AUDIENCE: &str = "switchboard-admin";

// ── JWT helper types and functions ────────────────────────────────────────────

/// Standard claims used by the admin JWT tests.
#[derive(Debug, Serialize, Deserialize)]
struct AdminTestClaims {
    sub: String,
    aud: String,
    iss: String,
    exp: i64,
    iat: i64,
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// Build the JWKS JSON that matches `TEST_RSA_PRIVATE_KEY_PKCS8`.
fn test_admin_jwks() -> String {
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

/// Sign an admin JWT with the test RSA private key.
fn sign_admin_jwt(claims: &AdminTestClaims) -> String {
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(TEST_KID.to_string());
    encode(
        &header,
        claims,
        &EncodingKey::from_rsa_pem(TEST_RSA_PRIVATE_KEY_PKCS8.as_bytes()).unwrap(),
    )
    .unwrap()
}

/// Start a wiremock server that serves the test JWKS at
/// `/.well-known/jwks.json` — the path used by `AdminAuthState::init_jwt`.
async fn start_admin_jwks_mock() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/.well-known/jwks.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(test_admin_jwks()),
        )
        .mount(&server)
        .await;
    server
}

/// Minimal valid TOML for a `ServerConfig` that works for reload tests.
///
/// The config sets up a static-key auth validator and an admin section with
/// `static_token` auth so the test harness's `test-api-key` is still accepted.
const MINIMAL_RELOAD_TOML: &str = r#"
[auth.validators.test]
type = "static_keys"
keys = ["test-api-key"]

[admin]
enabled = true
auth = "static_token"
static_token = "test-api-key"
"#;

// ── Tests ─────────────────────────────────────────────────────────────────────

/// `GET /admin/api/v1/health` must return 200 with `{"status":"ok"}`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_health_returns_ok() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let resp = harness.client.admin_get("health").await;

    assert_eq!(resp.status().as_u16(), 200, "admin /health must return 200");

    let body: serde_json::Value = resp.json().await.expect("admin /health must return JSON");
    assert_eq!(
        body["status"], "ok",
        "admin /health body must have status=ok"
    );
}

/// `GET /admin/api/v1/providers` must return 200 and a JSON array that
/// contains an entry with `"id": "openai"` (the provider configured in the
/// harness).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_providers_lists_openai() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let resp = harness.client.admin_get("providers").await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "admin /providers must return 200"
    );

    let body: serde_json::Value = resp
        .json()
        .await
        .expect("admin /providers must return JSON");
    assert!(
        body.is_array(),
        "admin /providers must return a JSON array, got: {body}"
    );

    let arr = body.as_array().unwrap();
    let has_openai = arr.iter().any(|entry| entry["id"] == "openai");
    assert!(
        has_openai,
        "admin /providers array must contain an entry with id=openai; got: {body}"
    );
}

/// `GET /admin/api/v1/config` must return 200 and a JSON object that contains
/// a `"providers"` key, confirming the full server config is serialised.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_config_endpoint() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let resp = harness.client.admin_get("config").await;

    assert_eq!(resp.status().as_u16(), 200, "admin /config must return 200");

    let body: serde_json::Value = resp.json().await.expect("admin /config must return JSON");
    assert!(
        body.is_object(),
        "admin /config must return a JSON object, got: {body}"
    );
    assert!(
        body.get("providers").is_some(),
        "admin /config JSON must contain a 'providers' key; got: {body}"
    );
}

/// `POST /admin/api/v1/config/reload` is expected to return 500 in the test
/// harness because no config file exists at the path used during test server
/// startup (an empty string `""`).  Verifying that the server returns a
/// response (rather than hanging or panicking) is the primary goal; the 500
/// status is an expected outcome in this environment.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_config_reload_returns_error() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let resp = harness.client.admin_post("config/reload", json!({})).await;

    // The server must respond (not hang).  In test mode the config path is ""
    // so the reload will fail — we just verify a non-2xx is returned.
    assert!(
        resp.status().as_u16() >= 400,
        "admin config/reload must return an error status when config file does not exist; got: {}",
        resp.status().as_u16()
    );
}

/// `GET /admin/api/v1/rate-limits` must return 200.  In the default test
/// harness there are no per-user overrides, so the response body may be an
/// empty array or an empty object — we only assert the HTTP status.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_rate_limits_endpoint() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let resp = harness.client.admin_get("rate-limits").await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "admin /rate-limits must return 200"
    );
}

/// Verify that protected admin endpoints require authentication.
///
/// `GET /admin/api/v1/health` is intentionally public (no auth required per
/// the handler doc-comment "no auth required").  This test uses
/// `GET /admin/api/v1/config` — a protected endpoint — to confirm that a
/// request without an `Authorization` header is rejected with 401.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_requires_auth() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let admin_base = harness
        .admin_addr
        .expect("admin_addr must be set when .with_admin() is used");

    let client = reqwest::Client::new();
    // Use /config — a protected endpoint — rather than /health which is public.
    let resp = client
        .get(format!("http://{}/admin/api/v1/config", admin_base))
        // No Authorization header.
        .send()
        .await
        .expect("request must not fail at the transport level");

    assert_eq!(
        resp.status().as_u16(),
        401,
        "admin /config must return 401 when no Authorization header is provided"
    );
}

/// Smoke-test that verifies the admin and proxy servers co-exist on separate
/// ports: a successful chat request through the proxy does not interfere with
/// admin endpoint availability.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_and_proxy_coexist() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "hello").mount(openai_mock).await;

    // Proxy request.
    let proxy_resp = harness
        .client
        .chat_completions(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .await;
    assert_eq!(proxy_resp.status().as_u16(), 200, "proxy must return 200");

    // Admin health — must still work after proxy traffic.
    let admin_resp = harness.client.admin_get("health").await;
    assert_eq!(
        admin_resp.status().as_u16(),
        200,
        "admin /health must still return 200 after proxy traffic"
    );
}

/// `POST /admin/api/v1/providers/openai/keys` must add a key to the pool and
/// return 201.  A subsequent `GET /admin/api/v1/providers/openai/keys` must
/// return a valid JSON array confirming the key pool is reachable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_add_and_list_provider_key() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    // POST a new key.  `type` is the serde-rename for `key_type`.
    let add_resp = harness
        .client
        .admin_post(
            "providers/openai/keys",
            json!({
                "id": "new-key",
                "type": "static",
                "api_key": "sk-new",
                "weight": 1.0
            }),
        )
        .await;

    let status = add_resp.status().as_u16();
    assert!(
        status == 200 || status == 201,
        "add provider key must return 200 or 201; got: {status}"
    );

    // List the keys for the openai provider.
    let list_resp = harness.client.admin_get("providers/openai/keys").await;
    assert_eq!(
        list_resp.status().as_u16(),
        200,
        "list provider keys must return 200"
    );

    let body: serde_json::Value = list_resp
        .json()
        .await
        .expect("list provider keys must return JSON");
    assert!(
        body.is_array(),
        "list provider keys must return a JSON array; got: {body}"
    );
}

/// `PUT /admin/api/v1/model-selection` must persist the new config and return
/// 200.  A subsequent `GET` must reflect the updated `mode` field.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_update_model_selection() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let put_resp = harness
        .client
        .admin_put(
            "model-selection",
            json!({
                "mode": "static",
                "model": "gpt-4o",
                "header": "x-model",
                "fallback": null,
                "mappings": {},
                "allowed_models": [],
                "overrides": {}
            }),
        )
        .await;

    assert_eq!(
        put_resp.status().as_u16(),
        200,
        "PUT model-selection must return 200"
    );

    // Read back and verify the mode was updated.
    let get_resp = harness.client.admin_get("model-selection").await;
    assert_eq!(
        get_resp.status().as_u16(),
        200,
        "GET model-selection must return 200"
    );

    let body: serde_json::Value = get_resp
        .json()
        .await
        .expect("GET model-selection must return JSON");
    assert_eq!(
        body["mode"], "static",
        "model-selection mode must be 'static' after PUT; got: {body}"
    );
}

/// `PUT /admin/api/v1/guardrails` must persist the new config and return 200.
/// A subsequent `GET` must reflect `enabled = true`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_update_guardrails() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let put_resp = harness
        .client
        .admin_put(
            "guardrails",
            json!({
                "enabled": true,
                "fail_mode": "open",
                "timeout": "500ms",
                "streaming_mode": "async_audit",
                "engines": []
            }),
        )
        .await;

    assert_eq!(
        put_resp.status().as_u16(),
        200,
        "PUT guardrails must return 200"
    );

    // Read back and verify enabled was updated.
    let get_resp = harness.client.admin_get("guardrails").await;
    assert_eq!(
        get_resp.status().as_u16(),
        200,
        "GET guardrails must return 200"
    );

    let body: serde_json::Value = get_resp
        .json()
        .await
        .expect("GET guardrails must return JSON");
    assert_eq!(
        body["enabled"], true,
        "guardrails enabled must be true after PUT; got: {body}"
    );
}

/// `PUT /admin/api/v1/rate-limits` must persist the new config and return 200.
/// A subsequent `GET` must reflect the updated `default_rpm`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_update_rate_limits() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let put_resp = harness
        .client
        .admin_put(
            "rate-limits",
            json!({
                "enabled": true,
                "default_rpm": 100,
                "default_tpm": 500000,
                "overrides": {}
            }),
        )
        .await;

    assert_eq!(
        put_resp.status().as_u16(),
        200,
        "PUT rate-limits must return 200"
    );

    // Read back and verify default_rpm was updated.
    let get_resp = harness.client.admin_get("rate-limits").await;
    assert_eq!(
        get_resp.status().as_u16(),
        200,
        "GET rate-limits must return 200"
    );

    let body: serde_json::Value = get_resp
        .json()
        .await
        .expect("GET rate-limits must return JSON");
    assert_eq!(
        body["default_rpm"], 100,
        "rate-limits default_rpm must be 100 after PUT; got: {body}"
    );
}

/// Full lifecycle for a per-entity rate limit override:
///
/// 1. `PUT  /admin/api/v1/rate-limits/overrides/alice` → 200
/// 2. `GET  /admin/api/v1/rate-limits/overrides`       → array must contain alice
/// 3. `DELETE /admin/api/v1/rate-limits/overrides/alice` → 200 or 204
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_rate_limit_override_lifecycle() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    // 1. Set the override.
    let put_resp = harness
        .client
        .admin_put(
            "rate-limits/overrides/alice",
            json!({"rpm": 50, "tpm": 100000}),
        )
        .await;
    assert_eq!(
        put_resp.status().as_u16(),
        200,
        "PUT rate-limits/overrides/alice must return 200"
    );

    // 2. List overrides — alice must be present.
    let list_resp = harness.client.admin_get("rate-limits/overrides").await;
    assert_eq!(
        list_resp.status().as_u16(),
        200,
        "GET rate-limits/overrides must return 200"
    );

    let body: serde_json::Value = list_resp
        .json()
        .await
        .expect("GET rate-limits/overrides must return JSON");
    let overrides = body["overrides"]
        .as_array()
        .expect("rate-limits/overrides response must have an 'overrides' array");
    let has_alice = overrides.iter().any(|entry| entry["id"] == "alice");
    assert!(
        has_alice,
        "overrides list must contain alice after PUT; got: {body}"
    );

    // 3. Delete the override.
    let del_resp = harness
        .client
        .admin_delete("rate-limits/overrides/alice")
        .await;
    let del_status = del_resp.status().as_u16();
    assert!(
        del_status == 200 || del_status == 204,
        "DELETE rate-limits/overrides/alice must return 200 or 204; got: {del_status}"
    );
}

/// `PUT /admin/api/v1/routing/semantic` must persist the config and return 200.
/// A subsequent `GET` must reflect `semantic.enabled = false`.
///
/// The `routing/semantic` API uses `RoutingConfig` which wraps
/// `SemanticRoutingConfig` under the `"semantic"` key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_semantic_routing_update() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let put_resp = harness
        .client
        .admin_put(
            "routing/semantic",
            json!({
                "semantic": {
                    "enabled": false,
                    "classifier": "heuristic",
                    "rules": [],
                    "default": {"preferred_models": []}
                }
            }),
        )
        .await;

    assert_eq!(
        put_resp.status().as_u16(),
        200,
        "PUT routing/semantic must return 200"
    );

    // Read back and verify enabled is false.
    let get_resp = harness.client.admin_get("routing/semantic").await;
    assert_eq!(
        get_resp.status().as_u16(),
        200,
        "GET routing/semantic must return 200"
    );

    let body: serde_json::Value = get_resp
        .json()
        .await
        .expect("GET routing/semantic must return JSON");
    assert_eq!(
        body["semantic"]["enabled"], false,
        "routing/semantic enabled must be false after PUT; got: {body}"
    );
}

// ── Part 1: Config hot-reload tests ──────────────────────────────────────────

/// `POST /admin/api/v1/config/reload` must return 200 when the server was
/// started with a valid config file path.
///
/// This test writes a minimal `ServerConfig` TOML to a temporary file,
/// passes the path to the harness via `.with_config_path(...)`, and then
/// calls the reload endpoint.  Unlike the existing
/// `test_e2e_admin_config_reload_returns_error` test (which passes `""` as
/// the path and therefore expects an error), this test verifies the happy
/// path where the file exists and is valid.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_config_reload_succeeds_with_valid_file() {
    // Write the minimal TOML to a temp file.  The file must remain alive for
    // the duration of the test so we keep the NamedTempFile in scope.
    let mut tmp = tempfile::NamedTempFile::new().expect("failed to create temp config file");
    tmp.write_all(MINIMAL_RELOAD_TOML.as_bytes())
        .expect("failed to write temp config file");

    let config_path = tmp
        .path()
        .to_str()
        .expect("temp file path is not valid UTF-8")
        .to_owned();

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .with_config_path(config_path)
        .build()
        .await;

    let resp = harness.client.admin_post("config/reload", json!({})).await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "config/reload must return 200 when config file is valid; got: {}",
        resp.status().as_u16()
    );
}

/// Hot-reload picks up changes written to the config file on disk.
///
/// 1. Write an initial TOML config (rate_limit enabled = false) to a temp file.
/// 2. Build the harness with that path.
/// 3. Verify rate-limits endpoint reflects the initial config (enabled = false).
/// 4. Overwrite the temp file with a new TOML that enables rate limiting and
///    sets a distinct `default_rpm`.
/// 5. Call `POST /admin/api/v1/config/reload` → expect 200.
/// 6. Call `GET /admin/api/v1/rate-limits` → verify `default_rpm` reflects
///    the updated value, confirming the hot-swap took effect.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_config_reload_new_settings_take_effect() {
    // Initial config: rate limiting disabled, default_rpm = 60 (the default).
    let initial_toml = r#"
[auth.validators.test]
type = "static_keys"
keys = ["test-api-key"]

[admin]
enabled = true
auth = "static_token"
static_token = "test-api-key"

[rate_limit]
enabled = false
default_rpm = 60
default_tpm = 100000
"#;

    let mut tmp = tempfile::NamedTempFile::new().expect("failed to create temp config file");
    tmp.write_all(initial_toml.as_bytes())
        .expect("failed to write initial config");

    let config_path = tmp
        .path()
        .to_str()
        .expect("temp file path is not valid UTF-8")
        .to_owned();

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .with_config_path(config_path.clone())
        .build()
        .await;

    // ── Step 3: initial rate-limits state ─────────────────────────────────────
    // The test harness's build_test_config sets rate_limit from .with_rate_limit()
    // (not from the TOML file), so we only check that the endpoint is reachable.
    let init_resp = harness.client.admin_get("rate-limits").await;
    assert_eq!(
        init_resp.status().as_u16(),
        200,
        "GET rate-limits must return 200 before reload"
    );

    // ── Step 4: write updated config with distinct default_rpm ────────────────
    let updated_toml = r#"
[auth.validators.test]
type = "static_keys"
keys = ["test-api-key"]

[admin]
enabled = true
auth = "static_token"
static_token = "test-api-key"

[rate_limit]
enabled = true
default_rpm = 999
default_tpm = 200000
"#;
    // Reopen the file for overwriting (seek + set_len, or use a fresh write).
    {
        let mut f = std::fs::File::create(&config_path)
            .expect("failed to reopen temp config file for writing");
        f.write_all(updated_toml.as_bytes())
            .expect("failed to write updated config");
    }

    // ── Step 5: trigger reload ────────────────────────────────────────────────
    let reload_resp = harness.client.admin_post("config/reload", json!({})).await;
    assert_eq!(
        reload_resp.status().as_u16(),
        200,
        "config/reload must return 200 after writing valid config; got: {}",
        reload_resp.status().as_u16()
    );

    // ── Step 6: verify the new default_rpm is live ────────────────────────────
    let rate_resp = harness.client.admin_get("rate-limits").await;
    assert_eq!(
        rate_resp.status().as_u16(),
        200,
        "GET rate-limits must return 200 after reload"
    );

    let body: serde_json::Value = rate_resp
        .json()
        .await
        .expect("GET rate-limits must return JSON");

    assert_eq!(
        body["default_rpm"], 999,
        "rate-limits default_rpm must be 999 after hot-reload; got: {body}"
    );
    assert_eq!(
        body["enabled"], true,
        "rate-limits enabled must be true after hot-reload; got: {body}"
    );
}

// ── Part 2: Admin JWT auth tests ──────────────────────────────────────────────

/// When the admin server is configured for JWT auth but the JWKS endpoint is
/// unreachable at startup, `init_jwt` fails (logged as a warning) and the
/// server continues running without a JWT validator.
///
/// A subsequent admin request with an arbitrary bearer token must be rejected
/// with 401 because:
/// - `config.admin.auth == "jwt"` → `validate_jwt` is called
/// - `jwt_validator` is `None` (init failed) → `AuthError::Config` → 401
///
/// This test also verifies that a server with a broken JWT issuer still starts
/// and serves the health endpoint (which is unauthenticated).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_jwt_auth_invalid_token_returns_401() {
    // Write a minimal config with JWT auth pointing to a permanently
    // unreachable issuer (port 1 refuses connections on all OSes).
    // The harness's build_test_config is bypassed here — we need a custom
    // admin config.  We achieve this by building the harness normally (which
    // sets static_token admin) and then making a raw request to an admin
    // server that was started with JWT auth.
    //
    // Simpler approach: start a dedicated server via start_test_server with
    // a custom ServerConfig that has admin.auth = "jwt".
    use crate::config::build_test_config;
    use crate::config::{BuildOpts, ProviderMocks};
    use switchboard_server::config::admin::AdminConfig;

    let proxy_port = crate::port_allocator::allocate();
    let admin_port = crate::port_allocator::allocate();
    let admin_listen = format!("127.0.0.1:{admin_port}");

    // Build a base config and then override the admin section.
    let openai_mock = MockServer::start().await;
    let mocks = ProviderMocks {
        openai: Some(openai_mock),
        anthropic: None,
        ollama: None,
        bedrock: None,
        vertex: None,
    };
    let mut cfg = build_test_config(
        &mocks,
        &BuildOpts {
            admin_enabled: true,
            admin_listen: admin_listen.clone(),
            ..BuildOpts::default()
        },
    );

    // Override admin to use JWT auth with an unreachable issuer.
    cfg.admin = AdminConfig {
        enabled: true,
        listen: admin_listen,
        auth: "jwt".into(),
        jwt_issuer: Some("http://127.0.0.1:1".into()), // unreachable → init_jwt fails
        jwt_audience: Some(ADMIN_JWT_AUDIENCE.into()),
        allowed_roles: vec![],
        static_token: None,
        mtls_ca: None,
    };

    // Start the server; init_jwt will warn and continue.
    let proxy_listener = tokio::net::TcpListener::bind(("127.0.0.1", proxy_port))
        .await
        .unwrap_or_else(|e| panic!("failed to bind proxy port: {e}"));
    let admin_listener = tokio::net::TcpListener::bind(("127.0.0.1", admin_port))
        .await
        .unwrap_or_else(|e| panic!("failed to bind admin port: {e}"));
    let proxy_addr = proxy_listener.local_addr().unwrap();
    let admin_addr = admin_listener.local_addr().unwrap();

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = switchboard_server::run_server(
            cfg,
            "",
            proxy_listener,
            Some(admin_listener),
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await;
    });

    // Wait for the proxy to be ready (it shares the same startup path).
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match client
            .get(format!("http://{proxy_addr}/health"))
            .header("Authorization", "Bearer test-api-key")
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => break,
            _ => {}
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("JWT-auth test server did not become ready within 5 seconds");
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // The admin health endpoint is public — it must return 200 even with JWT
    // auth configured (health is exempt from auth).
    let health_resp = client
        .get(format!("http://{admin_addr}/admin/api/v1/health"))
        .send()
        .await
        .expect("admin /health request must not fail");
    assert_eq!(
        health_resp.status().as_u16(),
        200,
        "admin /health must return 200 regardless of JWT config"
    );

    // A protected endpoint with a random bogus bearer token must be rejected.
    let protected_resp = client
        .get(format!("http://{admin_addr}/admin/api/v1/config"))
        .header("Authorization", "Bearer not-a-valid-jwt")
        .send()
        .await
        .expect("admin /config request must not fail at transport level");

    assert_eq!(
        protected_resp.status().as_u16(),
        401,
        "admin /config with invalid JWT token must return 401; \
         JWT validator was never initialised so any token is rejected"
    );

    // Clean up.
    let _ = shutdown_tx.send(());
}

/// When the admin JWT auth is fully configured — with a live JWKS mock and a
/// properly signed RS256 token — a request with a valid bearer JWT must
/// be accepted (200) and a request without a token must be rejected (401).
///
/// Flow:
/// 1. Start a wiremock JWKS server **before** building the harness so that
///    `init_jwt()` can fetch the JWKS during server startup.
/// 2. Build a custom ServerConfig with `admin.auth = "jwt"` pointing at the
///    wiremock issuer.
/// 3. Sign a JWT using the matching RSA private key.
/// 4. `GET /admin/api/v1/providers` with `Authorization: Bearer <signed_jwt>`
///    → expect 200.
/// 5. Same endpoint without a token → expect 401.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_admin_jwt_auth_with_valid_jwks() {
    use crate::config::build_test_config;
    use crate::config::{BuildOpts, ProviderMocks};
    use switchboard_server::config::admin::AdminConfig;

    // ── Step 1: start the JWKS mock server ────────────────────────────────────
    // Must be started BEFORE `run_server` so that `init_jwt()` can fetch keys.
    let jwks_server = start_admin_jwks_mock().await;
    let jwt_issuer = jwks_server.uri(); // e.g. "http://127.0.0.1:XXXXX"

    // ── Step 2: build a ServerConfig with JWT admin auth ─────────────────────
    let proxy_port = crate::port_allocator::allocate();
    let admin_port = crate::port_allocator::allocate();
    let admin_listen = format!("127.0.0.1:{admin_port}");

    let openai_mock = MockServer::start().await;
    let mocks = ProviderMocks {
        openai: Some(openai_mock),
        anthropic: None,
        ollama: None,
        bedrock: None,
        vertex: None,
    };
    let mut cfg = build_test_config(
        &mocks,
        &BuildOpts {
            admin_enabled: true,
            admin_listen: admin_listen.clone(),
            ..BuildOpts::default()
        },
    );

    cfg.admin = AdminConfig {
        enabled: true,
        listen: admin_listen,
        auth: "jwt".into(),
        jwt_issuer: Some(jwt_issuer.clone()),
        jwt_audience: Some(ADMIN_JWT_AUDIENCE.into()),
        allowed_roles: vec![],
        static_token: None,
        mtls_ca: None,
    };

    let proxy_listener = tokio::net::TcpListener::bind(("127.0.0.1", proxy_port))
        .await
        .unwrap_or_else(|e| panic!("failed to bind proxy port: {e}"));
    let admin_listener = tokio::net::TcpListener::bind(("127.0.0.1", admin_port))
        .await
        .unwrap_or_else(|e| panic!("failed to bind admin port: {e}"));
    let proxy_addr = proxy_listener.local_addr().unwrap();
    let admin_addr = admin_listener.local_addr().unwrap();

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = switchboard_server::run_server(
            cfg,
            "",
            proxy_listener,
            Some(admin_listener),
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await;
    });

    // Wait for the proxy server to be ready.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match client
            .get(format!("http://{proxy_addr}/health"))
            .header("Authorization", "Bearer test-api-key")
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => break,
            _ => {}
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("JWT-auth test server did not become ready within 5 seconds");
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // ── Step 3: sign a valid JWT ───────────────────────────────────────────────
    let claims = AdminTestClaims {
        sub: "admin-user@example.com".into(),
        aud: ADMIN_JWT_AUDIENCE.into(),
        iss: jwt_issuer.clone(),
        exp: now_secs() + 3600,
        iat: now_secs(),
    };
    let token = sign_admin_jwt(&claims);

    // ── Step 4: request with valid JWT → 200 ─────────────────────────────────
    let authed_resp = client
        .get(format!("http://{admin_addr}/admin/api/v1/providers"))
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .await
        .expect("admin /providers request with valid JWT must not fail");

    assert_eq!(
        authed_resp.status().as_u16(),
        200,
        "admin /providers must return 200 when a valid RS256 JWT is presented; \
         got: {}",
        authed_resp.status().as_u16()
    );

    // ── Step 5: request without token → 401 ──────────────────────────────────
    let no_auth_resp = client
        .get(format!("http://{admin_addr}/admin/api/v1/providers"))
        // No Authorization header.
        .send()
        .await
        .expect("admin /providers request without token must not fail at transport level");

    assert_eq!(
        no_auth_resp.status().as_u16(),
        401,
        "admin /providers must return 401 when no Authorization header is provided \
         (JWT auth mode); got: {}",
        no_auth_resp.status().as_u16()
    );

    // Clean up.
    let _ = shutdown_tx.send(());
}
