//! E2E scenario: key selector strategies.
//!
//! Verifies that the OpenAI key pool correctly rotates through multiple keys
//! according to the configured selector strategy.  The upstream mock records
//! every received request, so tests can inspect the `Authorization` header to
//! determine which key was forwarded.
//!
//! The `with_openai_keys` builder method replaces the default single-key pool
//! with a multi-key pool using the specified selector strategy.  The server
//! builds `Authorization: Bearer <api_key_value>` for each `"static"` key
//! entry in the pool and forwards it to the upstream provider.

use serde_json::json;

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai::mock_chat_ok;

// ── Shared chat request body ───────────────────────────────────────────────

fn chat_body() -> serde_json::Value {
    json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────

/// Round-robin selector alternates through keys in order over consecutive
/// requests.
///
/// With two keys `sk-key-a` and `sk-key-b`, the first request should use
/// `sk-key-a` and the second should use `sk-key-b` (or vice-versa), and both
/// keys must appear exactly once across the two requests.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_round_robin_alternates_keys() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_openai_keys(
            vec![("key-a", "sk-key-a"), ("key-b", "sk-key-b")],
            "round_robin",
        )
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Mount without an upper-call-count limit so both requests reach the mock.
    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Make two sequential requests so round-robin has a chance to rotate.
    let resp1 = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        resp1.status().as_u16(),
        200,
        "first request must return 200"
    );

    let resp2 = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        resp2.status().as_u16(),
        200,
        "second request must return 200"
    );

    // Inspect which Authorization headers the upstream mock received.
    let received = openai_mock
        .received_requests()
        .await
        .expect("failed to fetch received requests");

    assert_eq!(
        received.len(),
        2,
        "upstream mock must have received exactly 2 requests"
    );

    // Collect the Authorization header values sent to the upstream.
    let auth_values: Vec<String> = received
        .iter()
        .filter_map(|req| {
            req.headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
        })
        .collect();

    assert_eq!(
        auth_values.len(),
        2,
        "both upstream requests must carry an Authorization header"
    );

    // Both keys must appear exactly once (round-robin alternates).
    assert!(
        auth_values.contains(&"Bearer sk-key-a".to_string()),
        "expected 'Bearer sk-key-a' in upstream auth headers, got: {:?}",
        auth_values
    );
    assert!(
        auth_values.contains(&"Bearer sk-key-b".to_string()),
        "expected 'Bearer sk-key-b' in upstream auth headers, got: {:?}",
        auth_values
    );
}

/// Weighted-random selector with a single key always routes to that key.
///
/// With only one key `sk-key-1`, every request — regardless of selection
/// strategy — must use that key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_weighted_random_uses_configured_keys() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_openai_keys(vec![("key-1", "sk-key-1")], "weighted_random")
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    let resp = harness.client.chat_completions(chat_body()).await;
    assert_eq!(resp.status().as_u16(), 200, "request must return 200");

    // Verify the upstream received exactly 1 request.
    let received = openai_mock
        .received_requests()
        .await
        .expect("failed to fetch received requests");

    assert_eq!(
        received.len(),
        1,
        "upstream mock must have received exactly 1 request"
    );

    // Verify the correct key was forwarded.
    let auth = received[0]
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    assert_eq!(
        auth, "Bearer sk-key-1",
        "upstream request must use the configured key 'sk-key-1', got: {:?}",
        auth
    );
}

/// With three keys and round-robin, three consecutive requests each use a
/// different key and all return 200.
///
/// This also validates that the key pool is set up correctly with more than
/// two keys.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_multiple_keys_all_requests_succeed() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_openai_keys(
            vec![
                ("key-1", "sk-key-1"),
                ("key-2", "sk-key-2"),
                ("key-3", "sk-key-3"),
            ],
            "round_robin",
        )
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // No call-count limit — the round-robin cycles through all three keys.
    mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

    // Make three requests — one full cycle through the round-robin pool.
    for i in 0..3 {
        let resp = harness.client.chat_completions(chat_body()).await;
        assert_eq!(resp.status().as_u16(), 200, "request {} must return 200", i);
    }

    // All three requests must have reached the upstream.
    let received = openai_mock
        .received_requests()
        .await
        .expect("failed to fetch received requests");

    assert_eq!(
        received.len(),
        3,
        "upstream mock must have received exactly 3 requests, got {}",
        received.len()
    );

    // Collect and sort the keys used — each of the three keys must appear
    // exactly once in a full round-robin cycle.
    let mut auth_values: Vec<String> = received
        .iter()
        .filter_map(|req| {
            req.headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
        })
        .collect();
    auth_values.sort();

    assert_eq!(
        auth_values,
        vec![
            "Bearer sk-key-1".to_string(),
            "Bearer sk-key-2".to_string(),
            "Bearer sk-key-3".to_string(),
        ],
        "all three keys must be used exactly once in a full round-robin cycle"
    );
}
