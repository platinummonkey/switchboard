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

use serde_json::json;

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai::mock_chat_ok;

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
