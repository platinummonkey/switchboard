//! E2E scenario: usage_e2e_test
//!
//! End-to-end tests for the `UsageTracker` exposed via the admin API.
//!
//! After chat requests flow through the proxy, `UsageTracker` increments its
//! `total_requests` counter.  The admin `GET /usage` endpoint serialises a
//! `UsageSnapshot` whose `total_requests` field reflects those increments.
//!
//! Key facts from the source:
//! - `UsageSnapshot::total_requests` → `u64`
//! - The tracker is an `Arc<UsageTracker>` shared between `AppState` (writes)
//!   and `AdminState` (reads).
//! - `GET /admin/api/v1/usage` returns the snapshot as a JSON object.
//! - Writes happen inside the proxy handler after the upstream responds, so
//!   the admin query must come *after* the response is received.

use serde_json::json;

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai::mock_chat_ok;

// ── Shared helper ─────────────────────────────────────────────────────────────

fn chat_body() -> serde_json::Value {
    json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Before any proxy request is made, the usage tracker must report
/// `total_requests == 0`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_usage_zero_before_requests() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let resp = harness.client.admin_get("usage").await;

    assert_eq!(resp.status().as_u16(), 200, "admin /usage must return 200");

    let body: serde_json::Value = resp.json().await.expect("admin /usage must return JSON");
    assert_eq!(
        body["total_requests"].as_u64().unwrap_or(u64::MAX),
        0,
        "total_requests must be 0 before any proxy request; got: {body}"
    );
}

/// After one successful chat request, the tracker must report
/// `total_requests >= 1`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_usage_increments_after_request() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Make a chat request through the proxy.
    let proxy_resp = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        proxy_resp.status().as_u16(),
        200,
        "proxy request must succeed before checking usage"
    );

    // Query usage — must show at least 1 recorded request.
    let usage_resp = harness.client.admin_get("usage").await;
    assert_eq!(
        usage_resp.status().as_u16(),
        200,
        "admin /usage must return 200"
    );

    let body: serde_json::Value = usage_resp
        .json()
        .await
        .expect("admin /usage must return JSON");
    let total = body["total_requests"].as_u64().unwrap_or(0);
    assert!(
        total >= 1,
        "total_requests must be >= 1 after one proxy request; got: {body}"
    );
}

/// After two successful chat requests, the tracker must report
/// `total_requests >= 2`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_usage_tracks_multiple_requests() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Mount mock with no explicit expectation — wiremock will answer any number
    // of matching requests.
    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Make two proxy requests.
    for i in 0..2 {
        let resp = harness.client.chat_completions(chat_body()).await;
        assert_eq!(
            resp.status().as_u16(),
            200,
            "proxy request {i} must succeed"
        );
    }

    // Check usage.
    let usage_resp = harness.client.admin_get("usage").await;
    assert_eq!(
        usage_resp.status().as_u16(),
        200,
        "admin /usage must return 200"
    );

    let body: serde_json::Value = usage_resp
        .json()
        .await
        .expect("admin /usage must return JSON");
    let total = body["total_requests"].as_u64().unwrap_or(0);
    assert!(
        total >= 2,
        "total_requests must be >= 2 after two proxy requests; got: {body}"
    );
}
