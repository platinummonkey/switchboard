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

/// Verifies that a per-user rate limit override applied via the admin API
/// takes effect immediately for subsequent requests.
///
/// # Scenario
///
/// 1. Start the server with a generous global rate limit (100 RPM) and admin
///    enabled.
/// 2. Confirm the first request succeeds under the global limit.
/// 3. Apply a tight override for `"anonymous"` via the admin API: RPM=1.
/// 4. Make a second request — the first slot in the new window was consumed by
///    the health-check poll(s) or by step 2, so the override must now cause a
///    429 within a bounded number of attempts.
///
/// # Why this tests the admin integration
///
/// `PUT /admin/api/v1/rate-limits/overrides/{id}` calls
/// `RateLimitHandle::set_override` on the live handle shared with the running
/// `RateLimitLayer`, so the change is visible to the next request without any
/// server restart.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_rate_limit_override_for_anonymous_user() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .with_rate_limit(100, 1_000_000)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Apply a very tight override: 1 RPM for the anonymous bucket.
    let override_resp = harness
        .client
        .admin_put(
            "rate-limits/overrides/anonymous",
            json!({"rpm": 1, "tpm": 1_000_000}),
        )
        .await;
    assert_eq!(
        override_resp.status().as_u16(),
        200,
        "admin PUT rate-limits/overrides/anonymous must return 200"
    );

    // Drive requests until we see a 429 (override of 1 RPM is active).
    // The first window slot may already be consumed by the health-check poll
    // that occurs during server startup, or by the first chat request below.
    // We allow up to 5 attempts to observe the 429.
    let max_attempts = 5;
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
        "expected a 429 within {max_attempts} requests after setting anonymous RPM override to 1"
    );
}

/// Verifies that the TPM enforcement semantics are consistent with the
/// implementation.
///
/// # Implementation contract (from rate_limit.rs)
///
/// The `check_and_record` call at request time passes `estimated_tokens = 0`
/// because the exact token count is only known after the LLM responds.  The
/// `record_tokens` helper is available to update the bucket post-response, but
/// it is NOT called in the proxy handler's critical path (it is marked
/// `#[allow(dead_code)]` in `RateLimitState`).
///
/// As a result, token-based limits configured via `.with_rate_limit(rpm, tpm)`
/// are enforced only on the *previously accumulated* token total in the
/// bucket.  Because `estimated_tokens` is always 0 at check time, the bucket's
/// token count never increases through normal proxy requests, and TPM limits
/// with `tpm > 0` will never trigger a 429 via the E2E path.
///
/// This test documents that behaviour and verifies that a server configured
/// with a tight TPM limit (`tpm = 1`) still allows requests to pass through,
/// confirming that no spurious TPM-based 429s are generated in the current
/// implementation.
///
/// When `record_tokens` is wired into the proxy handler in a future phase,
/// this test should be updated to verify that the TPM limit is enforced after
/// sufficient token accumulation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_tpm_rate_limit_not_enforced_without_record_tokens() {
    // Configure with rpm=1000 (very generous) and tpm=1 (very tight).
    // Since estimated_tokens=0 at request time and record_tokens is not
    // called in the E2E proxy path, the tpm=1 limit should never trigger.
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_rate_limit(1_000, 1) // tpm=1 would block if tokens were recorded
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    // mock_chat_ok returns usage.total_tokens=15 but record_tokens is not
    // called in the proxy, so the bucket stays at 0 tokens.
    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Three consecutive requests must all pass despite tpm=1.
    // The check `bucket.tokens + 0 > 1` is false (0 > 1 is false), so no TPM
    // rejection occurs.
    for i in 0..3 {
        let resp = harness.client.chat_completions(chat_body()).await;
        assert_eq!(
            resp.status().as_u16(),
            200,
            "request {i} must return 200: TPM is not enforced when estimated_tokens=0 \
             and record_tokens is not called in the proxy path"
        );
    }
}
