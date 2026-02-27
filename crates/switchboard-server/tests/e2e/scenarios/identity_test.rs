//! E2E scenario: identity_test
//!
//! End-to-end tests for the identity resolution pipeline.
//!
//! The server runs an `IdentityChain` that tries resolvers in order:
//! 1. `ToolSpecificResolver` — reads `x-switchboard-tool` header
//! 2. `HeaderResolver` — reads `x-switchboard-user` header
//! 3. `JwtClaimResolver` — extracts user from JWT claim (requires JWT auth)
//! 4. `ApiKeyMappingResolver` — maps API keys to user IDs
//!
//! These tests verify that identity headers are accepted by the server and
//! do not break routing.  The resolved identity flows into `RequestContext`
//! which is consumed by `UsageTracker`, `RateLimitLayer`, and key pool
//! `select()`.

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

/// Sending `x-switchboard-user: alice` must not break routing.
///
/// The `HeaderResolver` extracts the user identity from this header and
/// stores it in `RequestContext.user_id`.  The upstream proxy request is
/// still forwarded normally, so the mock must return 200.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_identity_user_header_accepted() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    let resp = harness
        .client
        .chat_with_extra_headers(chat_body(), &[("x-switchboard-user", "alice")])
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "x-switchboard-user header must not break routing; expected 200"
    );
}

/// Sending `x-switchboard-tool: cursor` must not break routing.
///
/// The `ToolSpecificResolver` is always the first resolver in the chain and
/// only enriches the identity (sets tool name) without overriding `user_id`.
/// The upstream mock must still receive and answer the request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_identity_tool_header_accepted() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    let resp = harness
        .client
        .chat_with_extra_headers(chat_body(), &[("x-switchboard-tool", "cursor")])
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "x-switchboard-tool header must not break routing; expected 200"
    );
}

/// Sending both `x-switchboard-user` and `x-switchboard-tool` together must
/// not break routing.
///
/// Both headers are valid Switchboard protocol headers; the identity chain
/// processes them in order and the request is forwarded unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_identity_user_and_tool_together() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    let resp = harness
        .client
        .chat_with_extra_headers(
            chat_body(),
            &[
                ("x-switchboard-user", "bob"),
                ("x-switchboard-tool", "vscode"),
            ],
        )
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "x-switchboard-user + x-switchboard-tool together must not break routing; expected 200"
    );
}

/// A request tagged with `x-switchboard-user: alice` must be tracked by the
/// `UsageTracker`.
///
/// After a successful proxied request the admin `GET /usage` endpoint must
/// report `total_requests >= 1`, proving that identity-tagged requests are
/// recorded in the shared usage tracker.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_identity_user_flows_into_usage() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Send one request tagged with a user identity.
    let proxy_resp = harness
        .client
        .chat_with_extra_headers(chat_body(), &[("x-switchboard-user", "alice")])
        .await;
    assert_eq!(
        proxy_resp.status().as_u16(),
        200,
        "proxy request must succeed before checking usage"
    );

    // Verify usage was recorded.
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
        "total_requests must be >= 1 after a user-tagged proxy request; got: {body}"
    );
}

/// Two different users making requests must both receive 200.
///
/// The identity chain assigns each request a user identity but does not
/// restrict access based on identity alone.  Both `alice` and `bob` must
/// be served successfully.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_identity_different_users_both_allowed() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Mount once — the mock will answer both requests.
    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Request from alice.
    let resp_alice = harness
        .client
        .chat_with_extra_headers(chat_body(), &[("x-switchboard-user", "alice")])
        .await;
    assert_eq!(
        resp_alice.status().as_u16(),
        200,
        "alice's request must return 200"
    );

    // Request from bob.
    let resp_bob = harness
        .client
        .chat_with_extra_headers(chat_body(), &[("x-switchboard-user", "bob")])
        .await;
    assert_eq!(
        resp_bob.status().as_u16(),
        200,
        "bob's request must return 200"
    );
}
