//! E2E scenario: failover_test
//!
//! End-to-end tests for upstream error propagation and request routing
//! observable behaviors.
//!
//! These tests verify what the proxy actually does when:
//! - A key pool has a single key and the upstream is healthy.
//! - The upstream returns error responses (500, 503).
//! - Multiple sequential requests arrive.
//!
//! NOTE: Automatic retry with a second key on failure is not guaranteed by the
//! proxy implementation.  These tests verify observable behavior only.

use serde_json::json;

use crate::assertions::assert_received_n;
use crate::harness::TestHarnessBuilder;
use crate::mocks::openai::{mock_chat_error, mock_chat_ok};

// ── Shared helper ─────────────────────────────────────────────────────────────

fn chat_body() -> serde_json::Value {
    json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// With a single healthy key in the pool, sequential requests should all
/// succeed.  This test verifies the baseline: the proxy routes through to
/// the mock and returns 200 for both requests.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_pool_with_multiple_keys_serves_requests() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Mount the mock with no upper limit so it handles both requests.
    mock_chat_ok("gpt-4o", "hello world")
        .mount(openai_mock)
        .await;

    // First request.
    let resp1 = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        resp1.status().as_u16(),
        200,
        "first request must return 200"
    );

    // Second request — same pool, same key, should still succeed.
    let resp2 = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        resp2.status().as_u16(),
        200,
        "second request must return 200"
    );
}

/// When the upstream returns 503, the proxy must propagate a non-200 status
/// to the client.  The error must not be silently swallowed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_upstream_error_returns_error_to_client() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_error(503, "Service Unavailable")
        .mount(openai_mock)
        .await;

    let resp = harness.client.chat_completions(chat_body()).await;

    assert_ne!(
        resp.status().as_u16(),
        200,
        "upstream 503 must not be returned to the client as 200"
    );
}

/// When the upstream returns 500, the proxy must propagate a non-200 status.
/// This covers the internal-server-error path distinct from 503.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_upstream_500_propagated() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_error(500, "Internal Server Error")
        .mount(openai_mock)
        .await;

    let resp = harness.client.chat_completions(chat_body()).await;

    assert_ne!(
        resp.status().as_u16(),
        200,
        "upstream 500 must not be returned to the client as 200"
    );
}

/// Make 3 sequential requests and verify that the mock server received
/// exactly 3 requests — confirming the proxy forwards every request and does
/// not de-duplicate or cache them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_mock_receives_all_requests() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "response").mount(openai_mock).await;

    for _ in 0..3 {
        let resp = harness.client.chat_completions(chat_body()).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    assert_received_n(openai_mock, 3).await;
}
