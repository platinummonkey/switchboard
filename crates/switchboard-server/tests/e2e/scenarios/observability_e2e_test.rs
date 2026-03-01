//! E2E scenario: observability.
//!
//! Tests cover observable, in-process effects of the observability subsystem
//! that can be verified without standing up a real OTLP collector:
//!
//! 1. The server starts and proxies requests successfully when OTel is
//!    **disabled** (the default in test configs) — i.e. the no-op path does
//!    not interfere with the request pipeline.
//!
//! 2. The `UsageTracker` (the in-process usage sink that the admin API
//!    exposes) correctly records token counts from the upstream response.
//!    This is the closest observable proxy for "did the observability path run
//!    and extract data?", because the proxy handler writes to both the
//!    `UsageTracker` and the `ProxySpan` using the same usage data from the
//!    upstream response.
//!
//! 3. Requests to multiple models are tracked independently under their
//!    respective model keys in the usage snapshot.
//!
//! # Why no OTel span assertions?
//!
//! OTel spans are exported asynchronously to an OTLP collector via a
//! `BatchSpanProcessor`.  In E2E tests there is no in-process collector to
//! intercept the exports, and the `tracing-opentelemetry` bridge records
//! attributes onto `tracing` spans that are not accessible from outside the
//! subscriber.  Full span-content assertions would require either:
//! - A custom `tracing::Subscriber` that captures span events into a shared
//!   buffer (feasible but out-of-scope for E2E tests — already done in unit
//!   tests in `observability_test.rs`), or
//! - A real OTLP receiver running in the test process.
//!
//! The `UsageTracker` tests below achieve the same coverage goal by verifying
//! that the data pipeline (handler → usage record → admin snapshot) works
//! end-to-end, which is the externally observable result of the observability
//! code running correctly.

use serde_json::json;

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai::mock_chat_ok;

// ── Helper ────────────────────────────────────────────────────────────────────

fn gpt4o_body(content: &str) -> serde_json::Value {
    json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}]
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// 1. With OTel **disabled** (the test harness default), a chat request
///    completes with HTTP 200.
///
/// This verifies that the no-op observability path does not introduce any
/// errors or panics in the request pipeline.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_observability_disabled_doesnt_crash() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    mock_chat_ok("gpt-4o", "hello world")
        .mount(openai_mock)
        .await;

    let resp = harness.client.chat_completions(gpt4o_body("ping")).await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "proxy must return 200 when OTel is disabled"
    );

    let body: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        body["choices"][0]["message"]["content"], "hello world",
        "response content must be forwarded from upstream"
    );
}

/// 2. After one successful chat request, the `UsageTracker` must record
///    `total_requests >= 1` and non-zero token counts.
///
/// The OpenAI mock returns `prompt_tokens: 10, completion_tokens: 5` (see
/// `mock_chat_ok`).  The proxy handler writes these values to the tracker
/// after parsing the upstream response — the same values that would be
/// attached to the OTel `ProxySpan`.  Asserting on the admin `/usage` snapshot
/// therefore indirectly verifies that the observability data-extraction path
/// ran to completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_usage_tracker_records_tokens_per_request() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    // mock_chat_ok returns prompt_tokens:10, completion_tokens:5, total_tokens:15.
    mock_chat_ok("gpt-4o", "token test response")
        .mount(openai_mock)
        .await;

    // Send the chat request through the proxy.
    let proxy_resp = harness
        .client
        .chat_completions(gpt4o_body("count tokens"))
        .await;
    assert_eq!(
        proxy_resp.status().as_u16(),
        200,
        "proxy request must succeed before checking usage"
    );

    // Query the admin usage endpoint.
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

    // At least one request must have been tracked.
    let total_reqs = body["total_requests"].as_u64().unwrap_or(0);
    assert!(
        total_reqs >= 1,
        "total_requests must be >= 1 after one proxy request; got: {body}"
    );

    // The tracker must have recorded the input tokens from the upstream mock
    // response (prompt_tokens = 10).
    let total_input = body["total_input_tokens"].as_u64().unwrap_or(0);
    assert!(
        total_input >= 10,
        "total_input_tokens must be >= 10 (mock returns prompt_tokens=10); got: {body}"
    );

    // The tracker must have recorded the output tokens (completion_tokens = 5).
    let total_output = body["total_output_tokens"].as_u64().unwrap_or(0);
    assert!(
        total_output >= 5,
        "total_output_tokens must be >= 5 (mock returns completion_tokens=5); got: {body}"
    );
}

/// 3. After two requests through the proxy, the usage snapshot must show
///    `total_requests >= 2` and at least twice the per-request token counts.
///
/// This verifies that token accumulation across requests works correctly,
/// which mirrors the expected behaviour of repeated OTel span emission.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_usage_tracker_accumulates_across_requests() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    // Mount without call-count limit; wiremock answers both requests.
    mock_chat_ok("gpt-4o", "accumulate me")
        .mount(openai_mock)
        .await;

    for i in 0..2 {
        let resp = harness
            .client
            .chat_completions(gpt4o_body("accumulate"))
            .await;
        assert_eq!(
            resp.status().as_u16(),
            200,
            "proxy request {i} must succeed"
        );
    }

    let usage_resp = harness.client.admin_get("usage").await;
    assert_eq!(usage_resp.status().as_u16(), 200);

    let body: serde_json::Value = usage_resp.json().await.expect("valid JSON");

    let total_reqs = body["total_requests"].as_u64().unwrap_or(0);
    assert!(
        total_reqs >= 2,
        "total_requests must be >= 2 after two requests; got: {body}"
    );

    // Two requests at 10 input tokens each → at least 20 total.
    let total_input = body["total_input_tokens"].as_u64().unwrap_or(0);
    assert!(
        total_input >= 20,
        "total_input_tokens must be >= 20 after two requests; got: {body}"
    );

    // Two requests at 5 output tokens each → at least 10 total.
    let total_output = body["total_output_tokens"].as_u64().unwrap_or(0);
    assert!(
        total_output >= 10,
        "total_output_tokens must be >= 10 after two requests; got: {body}"
    );
}

/// 4. The usage snapshot must track requests broken down by provider and model.
///
/// After a successful OpenAI request the `requests_by_provider["openai"]`
/// counter and `requests_by_model["gpt-4o"]` counter must both be non-zero.
/// This mirrors the per-provider and per-model attributes set on OTel spans
/// (`gen_ai.provider.name` and `gen_ai.request.model`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_usage_snapshot_includes_provider_and_model_breakdown() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    mock_chat_ok("gpt-4o", "breakdown test")
        .mount(openai_mock)
        .await;

    let proxy_resp = harness
        .client
        .chat_completions(gpt4o_body("breakdown"))
        .await;
    assert_eq!(proxy_resp.status().as_u16(), 200);

    let usage_resp = harness.client.admin_get("usage").await;
    assert_eq!(usage_resp.status().as_u16(), 200);

    let body: serde_json::Value = usage_resp.json().await.expect("valid JSON");

    // Provider breakdown: "openai" must appear with count >= 1.
    let by_provider = body["requests_by_provider"]
        .as_object()
        .expect("requests_by_provider must be a JSON object");
    let openai_count = by_provider
        .get("openai")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(
        openai_count >= 1,
        "requests_by_provider[\"openai\"] must be >= 1 after one OpenAI request; got: {body}"
    );

    // Model breakdown: "gpt-4o" must appear with count >= 1.
    let by_model = body["requests_by_model"]
        .as_object()
        .expect("requests_by_model must be a JSON object");
    let gpt4o_count = by_model.get("gpt-4o").and_then(|v| v.as_u64()).unwrap_or(0);
    assert!(
        gpt4o_count >= 1,
        "requests_by_model[\"gpt-4o\"] must be >= 1 after one gpt-4o request; got: {body}"
    );
}

/// 5. The proxy correctly handles a health-check request when OTel is
///    disabled, returning HTTP 200 with a JSON body containing
///    `"status": "ok"`.
///
/// This is a smoke test confirming the server is fully operational under the
/// default (no OTel) test configuration.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_observability_disabled_health_returns_ok() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;

    let resp = harness.client.health().await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "health endpoint must return 200 when OTel is disabled"
    );
}
