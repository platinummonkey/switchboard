//! E2E scenario: rate_limit_test
//!
//! End-to-end tests for the Tower `RateLimitLayer`.
//!
//! The layer uses a fixed-window counter per user.  When the window is not yet
//! exhausted, requests pass through; once the per-minute request count reaches
//! the configured RPM limit, subsequent requests within the same window receive
//! 429 Too Many Requests.
//!
//! Key facts from the implementation (middleware/rate_limit.rs):
//! - User identity comes from the `ValidatedClient` extension set by AuthLayer.
//!   The static-key validator sets `user_id = None`, so the rate limiter falls
//!   back to the synthetic key `"anonymous"`.
//! - A new bucket is created on first access; the window is 60 seconds.
//! - `check_and_record` increments *before* forwarding, so RPM=1 means the
//!   second request within the same window is always rejected.

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

/// With a generous rate limit (100 RPM), a single request must be allowed
/// through and return 200.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_rate_limit_first_request_allowed() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_rate_limit(100, 1_000_000)
        .build()
        .await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    let resp = harness.client.chat_completions(chat_body()).await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "first request under a high rate limit must return 200"
    );
}

/// Verifies that once the RPM limit is exhausted, subsequent requests in the
/// same window receive 429.
///
/// The server startup helper (`wait_for_ready`) polls `/health` with
/// `"test-api-key"`, which counts against the `"anonymous"` bucket.  To
/// account for those health-check requests we use a low but non-trivial RPM
/// (e.g. 3) and drive the bucket to exhaustion by making additional
/// `chat_completions` calls until a 429 is observed.  We then assert the 429
/// was returned within a bounded number of attempts.
///
/// This approach is robust regardless of how many health-check polls occur
/// before the test body runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_rate_limit_exceeded_returns_429() {
    // Use RPM=3 so the limit is tight enough to hit within a handful of
    // requests but generous enough that normal startup health polls don't
    // immediately exhaust the budget.
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_rate_limit(3, 10_000)
        .build()
        .await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Mount with no upper bound — the rate limiter will reject excess requests
    // before they reach the mock, so the mock only needs to answer up to 3.
    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Drive requests until we hit a 429 or exhaust our attempt budget.
    let max_attempts = 10;
    let mut saw_429 = false;
    for i in 0..max_attempts {
        let resp = harness.client.chat_completions(chat_body()).await;
        let status = resp.status().as_u16();
        if status == 429 {
            saw_429 = true;
            break;
        }
        assert_eq!(
            status, 200,
            "expected 200 or 429, got {status} on attempt {i}"
        );
    }

    assert!(
        saw_429,
        "expected at least one 429 within {max_attempts} requests with RPM=3"
    );
}

/// When no rate limit is configured (the default has `enabled: false`), all
/// requests must pass through regardless of how many are sent in rapid
/// succession.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_no_rate_limit_by_default() {
    // Build WITHOUT calling .with_rate_limit() so the default
    // (RateLimitConfig { enabled: false, ... }) is used.
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Three rapid sequential requests must all succeed.
    for i in 0..3 {
        let resp = harness.client.chat_completions(chat_body()).await;
        assert_eq!(
            resp.status().as_u16(),
            200,
            "request {i} must return 200 when no rate limit is configured"
        );
    }
}
