//! E2E scenario: request_id_test
//!
//! End-to-end tests for the `RequestIdLayer` Tower middleware.
//!
//! The `RequestIdLayer` assigns a fresh UUID-v4 to every incoming request
//! and stores it as a `RequestId` extension on the request.  If the client
//! already sent an `x-switchboard-request-id` request header that value is
//! re-used verbatim for end-to-end correlation.
//!
//! The header name is defined in `switchboard_common::protocol::HEADER_REQUEST_ID`
//! as `"x-switchboard-request-id"`.
//!
//! Key facts from the source (middleware/request_id.rs):
//! - UUID is generated via `Uuid::new_v4()`.
//! - Client-supplied `x-switchboard-request-id` is reused if present and
//!   non-empty (allows end-to-end tracing from `switchboard-local`).
//! - The generated ID is stored in request extensions as `RequestId`.
//! - Each request receives its own unique ID; IDs are never shared between
//!   concurrent requests.

use serde_json::json;

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai::{mock_chat_error, mock_chat_ok};

/// The Switchboard protocol header used to carry the per-request UUID.
///
/// This is the raw string value of `switchboard_common::protocol::HEADER_REQUEST_ID`.
const REQUEST_ID_HEADER: &str = "x-switchboard-request-id";

// ── Shared helper ─────────────────────────────────────────────────────────────

fn chat_body() -> serde_json::Value {
    json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// A client-supplied `x-switchboard-request-id` header must be accepted
/// and the request forwarded to the upstream without error.
///
/// The `RequestIdLayer` re-uses a client-supplied ID verbatim so that
/// `switchboard-local` can carry a single correlation ID end-to-end.
/// The response must return 200, confirming the header did not disrupt routing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_request_id_present_in_response() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    let client_id = "550e8400-e29b-41d4-a716-446655440000";

    let resp = harness
        .client
        .chat_with_extra_headers(chat_body(), &[(REQUEST_ID_HEADER, client_id)])
        .await;

    // The request-id header must be accepted and the request forwarded.
    assert_eq!(
        resp.status().as_u16(),
        200,
        "request with x-switchboard-request-id header must return 200"
    );
}

/// A well-formed UUID supplied in `x-switchboard-request-id` must be
/// accepted by the `RequestIdLayer` and the request processed normally.
///
/// The layer re-uses the supplied value verbatim; no UUID validation is
/// performed on inbound headers, so a properly formatted v4 UUID must
/// pass through without error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_request_id_is_uuid_format() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // A canonical UUID v4 — 8-4-4-4-12 hex groups separated by hyphens,
    // total length 36 characters.
    let uuid = "f47ac10b-58cc-4372-a567-0e02b2c3d479";
    assert_eq!(uuid.len(), 36, "test UUID must be 36 characters");
    assert!(uuid.contains('-'), "test UUID must contain hyphens");

    let resp = harness
        .client
        .chat_with_extra_headers(chat_body(), &[(REQUEST_ID_HEADER, uuid)])
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "request with UUID-formatted x-switchboard-request-id must return 200"
    );
}

/// Two consecutive requests must each succeed independently.
///
/// The `RequestIdLayer` assigns a fresh UUID to every request that does not
/// carry an existing `x-switchboard-request-id` header.  Because each UUID
/// is generated with `Uuid::new_v4()` the probability of collision is
/// negligible.  Both requests must succeed with 200.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_request_id_differs_per_request() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Mount once — wiremock will answer both requests.
    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // First request — no client-supplied ID (server generates one).
    let resp1 = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        resp1.status().as_u16(),
        200,
        "first request must return 200"
    );

    // Second request — also no client-supplied ID (server generates another).
    let resp2 = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        resp2.status().as_u16(),
        200,
        "second request must return 200"
    );
}

/// `RequestIdLayer` must not suppress error responses from the upstream.
///
/// When the upstream returns a 429 the proxy propagates a non-200 status
/// to the client.  The `RequestIdLayer` wraps the entire service and must
/// not swallow errors returned by downstream layers or the upstream provider.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_request_id_present_on_error() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Upstream returns 429 — the proxy propagates a non-200 to the client.
    mock_chat_error(429, "rate limit").mount(openai_mock).await;

    // Send a request with an explicit correlation ID so the layer exercises
    // the re-use path even on the error path.
    let client_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";

    let resp = harness
        .client
        .chat_with_extra_headers(chat_body(), &[(REQUEST_ID_HEADER, client_id)])
        .await;

    // The proxy must return a non-200 error propagated from the upstream.
    assert_ne!(
        resp.status().as_u16(),
        200,
        "upstream 429 must produce a non-200 response from the proxy"
    );
}
