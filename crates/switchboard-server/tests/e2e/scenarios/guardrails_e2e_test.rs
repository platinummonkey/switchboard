//! E2E scenario: Guardrail pipeline integrated into the full proxy stack.
//!
//! Tests cover:
//! - A request containing a blocked keyword is rejected with 403 Forbidden
//! - A clean request passes through and returns the upstream response
//! - The upstream provider is never called when the guardrail blocks a request
//!
//! The `with_keyword_guardrail` builder method configures a `builtin_keyword`
//! pre-request engine that blocks any request whose message content contains
//! one of the specified keywords. When blocked, the middleware returns
//! `403 Forbidden` with a JSON body `{"error":"forbidden","message":"..."}`.

use serde_json::json;

use crate::assertions::assert_received_n;
use crate::harness::TestHarnessBuilder;
use crate::mocks::openai;

/// 1. A request whose message content contains the blocked keyword is rejected
///    with HTTP 403 Forbidden by the guardrail middleware.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_guardrail_blocks_keyword_request() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_keyword_guardrail(vec!["FORBIDDEN".into()])
        .build()
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Please do something FORBIDDEN for me."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        403,
        "request containing the blocked keyword must be rejected with 403 Forbidden"
    );
}

/// 2. A clean request that does not contain any blocked keyword passes through
///    the guardrail and returns the upstream OpenAI response with 200 OK.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_guardrail_allows_clean_request() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_keyword_guardrail(vec!["FORBIDDEN".into()])
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    openai::mock_chat_ok("gpt-4o", "Paris is the capital of France.")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the capital of France?"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "clean request must pass with 200 OK"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["choices"][0]["message"]["content"], "Paris is the capital of France.",
        "response content should match the mock"
    );
}

/// 3. When the guardrail blocks a request, the upstream provider must never
///    receive the request — the pipeline short-circuits before forwarding.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_guardrail_blocked_does_not_reach_provider() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_keyword_guardrail(vec!["FORBIDDEN".into()])
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    // Mount a mock so we can count calls; it should never fire.
    openai::mock_chat_ok("gpt-4o", "should not be reached")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "This message contains FORBIDDEN content."}]
    });

    harness.client.chat_completions(body).await;

    // The upstream mock must have received zero requests.
    assert_received_n(mock_server, 0).await;
}
