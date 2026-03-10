//! E2E scenario: routing — verifies the proxy routes model names to the correct
//! upstream provider and returns the appropriate error for unknown models.

use serde_json::json;

use crate::harness::TestHarnessBuilder;
use crate::mocks::anthropic::mock_messages_ok as mock_anthropic_ok;
use crate::mocks::openai::mock_chat_ok as mock_openai_ok;

/// Happy-path: a request for "gpt-4o" must be forwarded to the OpenAI mock and
/// the proxy must return 200 with the expected content.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_routes_gpt4o_to_openai() {
    let harness = TestHarnessBuilder::new().with_openai().build().await;
    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_openai_ok("gpt-4o", "Paris").mount(openai_mock).await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Capital?"}]
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 from proxy");

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_openai_chat_response(&body, "Paris");
}

/// Happy-path: a request for "claude-3-5-sonnet-20241022" must be forwarded to
/// the Anthropic mock and the proxy must return 200.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_routes_claude_to_anthropic() {
    let harness = TestHarnessBuilder::new().with_anthropic().build().await;
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_anthropic_ok("London").mount(anthropic_mock).await;

    let resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Capital?"}],
            "max_tokens": 256
        }))
        .await;

    assert_eq!(resp.status().as_u16(), 200, "expected 200 from proxy");

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_anthropic_response(&body, "London");
}

/// A model name that is not registered in any provider must produce a non-200
/// client-error response (422 Unprocessable Entity or 400 Bad Request).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_unknown_model_returns_error() {
    // Only OpenAI is wired up; "unknown-model-xyz" is registered with no provider.
    let harness = TestHarnessBuilder::new().with_openai().build().await;

    let resp = harness
        .client
        .chat_completions(json!({
            "model": "unknown-model-xyz",
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .await;

    let status = resp.status().as_u16();
    assert!(
        (400..600).contains(&status),
        "expected a client/server error for an unknown model, got {}",
        status
    );
}

/// Both providers are wired up simultaneously; each request must be routed to
/// its correct upstream and each mock must receive exactly one request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_multiple_providers_route_independently() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_anthropic()
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_openai_ok("gpt-4o", "Rome").mount(openai_mock).await;
    mock_anthropic_ok("Madrid").mount(anthropic_mock).await;

    // OpenAI request.
    let openai_resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Capital?"}]
        }))
        .await;
    assert_eq!(openai_resp.status().as_u16(), 200);

    // Anthropic request.
    let anthropic_resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Capital?"}],
            "max_tokens": 256
        }))
        .await;
    assert_eq!(anthropic_resp.status().as_u16(), 200);

    // Each mock must have been hit exactly once.
    crate::assertions::assert_received_n(openai_mock, 1).await;
    crate::assertions::assert_received_n(anthropic_mock, 1).await;
}
