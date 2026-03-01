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

/// Verifies that TPM enforcement works end-to-end now that `record_tokens` is
/// called in the proxy handler after each response.
///
/// # Scenario
///
/// The mock returns `usage.total_tokens = 15` for every request.
/// `mock_chat_ok` returns `prompt_tokens=10, completion_tokens=5, total_tokens=15`.
/// The proxy records `input_tokens + output_tokens = 10 + 5 = 15` tokens
/// post-response via `RateLimitHandle::record_tokens`.
///
/// With `rpm = 1000` (generous) and `tpm = 20`:
///
/// - Request 1: pre-check sees 0 tokens accumulated → passes.
///   Post-response: bucket = 15 tokens.
/// - Request 2: pre-check sees 15 tokens + 0 estimated = 15 ≤ 20 → passes.
///   Post-response: bucket = 30 tokens.
/// - Request 3: pre-check sees 30 tokens + 0 estimated = 30 > 20 → **429**.
///
/// We drive requests until a 429 is observed within a bounded number of
/// attempts (to tolerate the startup health-check polls consuming some RPM
/// slots, and to handle any additional health polls before our requests run).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_tpm_rate_limit_enforced_after_record_tokens() {
    // rpm=1000 so RPM is never the bottleneck; tpm=20 so two requests
    // (each consuming 15 tokens) should exhaust the token budget.
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_rate_limit(1_000, 20)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    // mock_chat_ok returns usage: prompt_tokens=10, completion_tokens=5 → 15 total.
    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Drive requests until we observe a 429 (TPM exhausted) or reach the
    // attempt budget.  Health-check polls during startup hit /health which has
    // no usage recording, so only proxy requests accumulate tokens.
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
        "expected a 429 within {max_attempts} requests: tpm=20 with 15-token responses \
         should exhaust the token budget after 2 successful requests"
    );
}
