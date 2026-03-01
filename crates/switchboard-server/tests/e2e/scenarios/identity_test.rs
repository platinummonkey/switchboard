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

// ── ApiKeyMappingResolver E2E tests ───────────────────────────────────────────
//
// `ApiKeyMappingResolver` maps incoming raw API keys to `UserIdentity` values.
// The auth middleware stores the raw key in `ctx.switchboard_headers["__api_key"]`
// and the resolver looks it up in a static mapping table provided at construction
// time.
//
// # Current wiring status
//
// `build_identity_chain()` in `handler.rs` handles the `"api_key"` entry from
// `IdentityConfig.resolvers` in a catch-all `_` arm that logs
// `"identity resolver not yet wired"` and skips the entry.  The resolver is
// therefore NOT active in the E2E server's identity chain.
//
// Additionally, `IdentityConfig` does not carry a mapping table; the
// `ApiKeyMappingResolver` takes its `HashMap<String, UserIdentity>` at
// construction time and the harness provides no builder API to inject one.
//
// # What these tests verify
//
// Because the resolver is not wired, the tests focus on the observable system
// invariants that must hold regardless:
//
// 1. Requests authenticated with the static test key succeed without any
//    identity header (identity falls back to "anonymous").
// 2. The `x-switchboard-user` header (HeaderResolver) takes priority over any
//    future api_key mapping entry — this is the correct chain ordering.
// 3. An invalid auth key is rejected by `AuthLayer` before the identity chain
//    runs, so `ApiKeyMappingResolver` is never invoked with an invalid key.
// 4. Multiple identity-tagged requests in the same server process do not
//    interfere with each other.
// 5. The `UsageTracker` still records requests that resolve to "anonymous"
//    identity (no identity header provided).

/// A request authenticated with the static test key succeeds even when no
/// `x-switchboard-user` header is present.
///
/// With the default identity chain (header > jwt > api_key), all resolvers
/// return `None` for this request — `HeaderResolver` sees no user header,
/// `JwtClaimResolver` sees no JWT claim, and `ApiKeyMappingResolver` is not
/// wired.  The chain falls back to `UserIdentity::anonymous()`.  The proxy
/// still forwards the request successfully.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_api_key_mapping_unknown_key_falls_through_to_anonymous() {
    // Default harness: single static key "test-api-key", no user header.
    // Even if ApiKeyMappingResolver were wired, "test-api-key" would not be
    // in an empty mapping table.  Expected outcome: identity resolves to
    // anonymous and the request is proxied successfully.
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "anon-ok").mount(openai_mock).await;

    // Send with only the auth key — no x-switchboard-user header.
    let resp = harness.client.chat_completions(chat_body()).await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "request with static key and no user header must succeed (anonymous identity)"
    );

    // The upstream received exactly one request — no early rejection.
    crate::assertions::assert_received_n(openai_mock, 1).await;
}

/// A request with a `x-switchboard-user` header overrides the identity that
/// would otherwise come from the `ApiKeyMappingResolver`.
///
/// When both sources are present the `HeaderResolver` fires first (it is
/// earlier in the chain than `api_key`), so `x-switchboard-user` wins.
/// This confirms that sending the user header does not confuse the chain even
/// when an api_key mapping resolver entry would also match (if wired).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_api_key_mapping_header_identity_takes_priority() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "header-wins")
        .mount(openai_mock)
        .await;

    // Provide both the static auth key (accepted by auth layer) and the
    // x-switchboard-user header (accepted by HeaderResolver).
    // The HeaderResolver fires before any api_key mapping resolver would, so
    // user_id in RequestContext is "alice@example.com".
    let resp = harness
        .client
        .chat_with_extra_headers(chat_body(), &[("x-switchboard-user", "alice@example.com")])
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "request with both static key and user header must succeed"
    );

    // Verify the upstream received the request (identity did not block routing).
    crate::assertions::assert_received_n(openai_mock, 1).await;
}

/// Sending an invalid API key is rejected by the auth layer before reaching
/// any identity resolver.
///
/// The `AuthLayer` validates credentials first; only after a successful auth
/// validation does the identity chain run.  A bad auth key returns 401 and the
/// `ApiKeyMappingResolver` never executes — there is no `__api_key` entry in
/// the context for an unauthenticated request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_api_key_mapping_invalid_auth_key_rejected_before_identity_chain() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;

    // No mock mounted — the request must be rejected before reaching upstream.

    // Use a key that is NOT in the static_keys validator pool.
    let resp = harness
        .client
        .chat_with_key(chat_body(), "sk-not-a-valid-key")
        .await;

    assert_eq!(
        resp.status().as_u16(),
        401,
        "request with invalid auth key must be rejected with 401 before identity resolution"
    );
}

/// Multiple requests with different identity headers in the same server
/// instance must all succeed and not interfere with each other.
///
/// Each handler invocation creates a fresh `IdentityChain` and `RequestContext`,
/// so concurrent identity-tagged requests resolve independently.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_api_key_mapping_multiple_identity_tagged_requests_succeed() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // A single mock that answers all requests regardless of model or identity.
    mock_chat_ok("gpt-4o", "multi-ok").mount(openai_mock).await;

    // Fire three requests in sequence with different identity headers.
    for user in ["alice@example.com", "bob@example.com", "carol@example.com"] {
        let resp = harness
            .client
            .chat_with_extra_headers(chat_body(), &[("x-switchboard-user", user)])
            .await;
        assert_eq!(
            resp.status().as_u16(),
            200,
            "request for user '{user}' must return 200"
        );
    }

    // Three upstream requests received — one per user.
    crate::assertions::assert_received_n(openai_mock, 3).await;
}

/// Usage tracking still works when a request has no explicit user identity.
///
/// When no `x-switchboard-user` header is present and the
/// `ApiKeyMappingResolver` is not wired, identity resolves to `"anonymous"`.
/// The `UsageTracker` must still record the request under that identity.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_api_key_mapping_anonymous_identity_recorded_in_usage() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .build()
        .await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "usage-anon")
        .mount(openai_mock)
        .await;

    // Request without any identity header — resolves to "anonymous".
    let proxy_resp = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        proxy_resp.status().as_u16(),
        200,
        "anonymous request must succeed"
    );

    // Admin usage endpoint must report at least one request.
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
        "total_requests must be >= 1 after anonymous proxy request; got: {body}"
    );
}
