//! E2E scenario: Guardrail pipeline integrated into the full proxy stack.
//!
//! Tests cover:
//! - A request containing a blocked keyword is rejected with 403 Forbidden
//! - A clean request passes through and returns the upstream response
//! - The upstream provider is never called when the guardrail blocks a request
//! - Regex guardrail blocks requests matching a pattern (SSN)
//! - Regex guardrail allows requests that do not match the pattern
//! - Token limit guardrail blocks requests that exceed the configured limit
//! - Token limit guardrail allows short requests
//! - Guardrail passthrough on streaming (clean streaming request passes)
//! - Multiple engines: first engine blocks, second is never evaluated
//!
//! The `with_keyword_guardrail` builder method configures a `builtin_keyword`
//! pre-request engine that blocks any request whose message content contains
//! one of the specified keywords. When blocked, the middleware returns
//! `403 Forbidden` with a JSON body `{"error":"forbidden","message":"..."}`.

use serde_json::json;
use switchboard_server::config::guardrails::{EngineConfig, GuardrailsConfig, RegexRule};

use crate::assertions::assert_received_n;
use crate::client::TestClient;
use crate::harness::TestHarnessBuilder;
use crate::mocks::openai;

// ── Config helpers ─────────────────────────────────────────────────────────────

/// Build a `GuardrailsConfig` with a single `builtin_regex` pre-request engine
/// that blocks content matching `pattern`.
fn regex_guardrail(pattern: &str) -> GuardrailsConfig {
    GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "builtin_regex".into(),
            phase: "pre_request".into(),
            rules: vec![RegexRule {
                name: "test-pattern".into(),
                pattern: pattern.into(),
                action: "block".into(),
            }],
            action: None,
            keywords: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            endpoint: None,
            timeout: None,
            tls: false,
            headers: Default::default(),
        }],
    }
}

/// Build a `GuardrailsConfig` with a single `builtin_token_limit` pre-request
/// engine that blocks requests whose estimated token count exceeds `max_input`.
fn token_limit_guardrail(max_input: u32) -> GuardrailsConfig {
    GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "builtin_token_limit".into(),
            phase: "pre_request".into(),
            max_input_tokens: Some(max_input),
            max_output_tokens: None,
            action: Some("block".into()),
            rules: vec![],
            keywords: vec![],
            endpoint: None,
            timeout: None,
            tls: false,
            headers: Default::default(),
        }],
    }
}

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

// ── Regex guardrail tests ──────────────────────────────────────────────────────

/// 4. A request whose message contains text matching the SSN regex pattern is
///    rejected with HTTP 403 Forbidden.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_regex_guardrail_blocks_matching_prompt() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(regex_guardrail(r"\b\d{3}-\d{2}-\d{4}\b"))
        .build()
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Please help me, my SSN is 123-45-6789."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        403,
        "request containing an SSN pattern must be rejected with 403 Forbidden"
    );
}

/// 5. A request whose message does NOT match the SSN regex pattern passes
///    through the guardrail and returns the upstream 200 OK response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_regex_guardrail_allows_non_matching() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(regex_guardrail(r"\b\d{3}-\d{2}-\d{4}\b"))
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    openai::mock_chat_ok("gpt-4o", "No sensitive data here.")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the weather like today?"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "clean request (no SSN) must pass with 200 OK"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["choices"][0]["message"]["content"], "No sensitive data here.",
        "response content should match the mock"
    );
}

// ── Token limit guardrail tests ────────────────────────────────────────────────

/// 6. A request whose message greatly exceeds the 5-token limit (heuristic:
///    1 token ≈ 4 chars; the long essay is ~100+ tokens) is rejected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_token_limit_blocks_large_request() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(token_limit_guardrail(5))
        .build()
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": "Write me a very long essay about the history of computing and all its milestones from the 1940s to today"
        }]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_ne!(
        resp.status().as_u16(),
        200,
        "request exceeding the token limit must not return 200"
    );
}

/// 7. A short request that comfortably fits within a generous 1000-token limit
///    passes through the guardrail and returns the upstream 200 OK response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_token_limit_allows_short_request() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(token_limit_guardrail(1000))
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    openai::mock_chat_ok("gpt-4o", "Short answer.")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "short request within token limit must pass with 200 OK"
    );
}

// ── Streaming passthrough test ─────────────────────────────────────────────────

/// 8. A clean streaming request passes through a keyword guardrail (keyword
///    "BLOCK" is not present) and the SSE response is delivered correctly.
///    The guardrail must not interfere with the streaming flow.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_guardrail_passthrough_streaming() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_keyword_guardrail(vec!["BLOCK".into()])
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");

    let chunk = r#"{"choices":[{"delta":{"content":"hello"},"index":0,"finish_reason":"stop"}]}"#;
    openai::mock_chat_streaming(&[chunk])
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Say hello please."}]
    });

    let resp = harness.client.chat_completions_stream(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "clean streaming request must pass with 200 OK"
    );

    let events = TestClient::collect_sse(resp).await;
    assert!(
        !events.is_empty(),
        "at least one SSE data event must be delivered; got none"
    );
    assert!(
        events.contains(&chunk.to_string()),
        "expected the streamed chunk in SSE events, got: {:?}",
        events
    );
}

// ── Multiple engines test ──────────────────────────────────────────────────────

/// 9. With two keyword engines configured, a request matching the first
///    engine's keyword is blocked with 403 before the second engine runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_multiple_guardrail_engines_first_blocks() {
    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![
            EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "pre_request".into(),
                action: Some("block".into()),
                keywords: vec!["FORBIDDEN".into()],
                rules: vec![],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
            EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "pre_request".into(),
                action: Some("block".into()),
                keywords: vec!["ALSO_BAD".into()],
                rules: vec![],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
        ],
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(cfg)
        .build()
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "This message contains FORBIDDEN content."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        403,
        "request matching the first engine keyword must be rejected with 403 Forbidden"
    );
}

// ── Post-response guardrail helpers ───────────────────────────────────────────

/// Build a `GuardrailsConfig` with a single `builtin_keyword` post-response
/// engine that blocks responses containing any of the given `keywords`.
fn post_response_keyword_guardrail(keywords: Vec<String>) -> GuardrailsConfig {
    GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "builtin_keyword".into(),
            phase: "post_response".into(), // ← post_response phase
            action: Some("block".into()),
            keywords,
            rules: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            endpoint: None,
            timeout: None,
            tls: false,
            headers: Default::default(),
        }],
    }
}

// ── Post-response guardrail tests ─────────────────────────────────────────────

/// 10. When the upstream returns a response whose content contains a blocked
///     keyword, the post-response guardrail intercepts and blocks it with 403.
///     The request itself was clean — only the response triggered the guardrail.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_post_response_guardrail_blocks_toxic_response() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(post_response_keyword_guardrail(vec![
            "TOXIC_CONTENT".into(),
        ]))
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    // The upstream returns a response containing the blocked keyword.
    openai::mock_chat_ok("gpt-4o", "This response contains TOXIC_CONTENT.")
        .mount(mock_server)
        .await;

    // The request itself is clean — no keyword in the prompt.
    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tell me something helpful."}]
    });

    let resp = harness.client.chat_completions(body).await;
    // The post-response guardrail must intercept the toxic response and block
    // it with 403 Forbidden. The upstream was called (request was clean) but
    // the response is suppressed.
    assert_eq!(
        resp.status().as_u16(),
        403,
        "response containing a blocked keyword must be rejected with 403 Forbidden"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["error"], "forbidden",
        "blocked response body must have error=forbidden"
    );
}

/// 11. When the upstream returns a clean response, the post-response guardrail
///     configured with a keyword that is absent from the response passes it
///     through with the original 200 OK status.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_post_response_guardrail_allows_clean_response() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(post_response_keyword_guardrail(vec![
            "TOXIC_CONTENT".into(),
        ]))
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    // The upstream returns a response that does NOT contain the blocked keyword.
    openai::mock_chat_ok("gpt-4o", "This is a safe and helpful response.")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tell me something helpful."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "clean response must pass through the post-response guardrail with 200 OK"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["choices"][0]["message"]["content"], "This is a safe and helpful response.",
        "clean response content must be forwarded unchanged"
    );
}

/// 12. Unlike pre-request blocking (which short-circuits before the upstream is
///     called), a post-response guardrail always calls the upstream first and
///     only evaluates the response it receives.  This test verifies that the
///     upstream mock receives exactly one request even though the response is
///     subsequently blocked by the guardrail.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_post_response_guardrail_upstream_still_called() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(post_response_keyword_guardrail(vec![
            "TOXIC_CONTENT".into(),
        ]))
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    openai::mock_chat_ok("gpt-4o", "Contains TOXIC_CONTENT here.")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tell me something."}]
    });

    harness.client.chat_completions(body).await;

    // The upstream must have received exactly one request.  This distinguishes
    // post-response blocking (upstream is always called) from pre-request
    // blocking (upstream is never called when blocked before forwarding).
    assert_received_n(mock_server, 1).await;
}

// ── Combined pre + post guardrail tests ───────────────────────────────────────

/// Build a `GuardrailsConfig` with two engines:
/// - A `builtin_keyword` pre-request engine blocking `pre_keyword`.
/// - A `builtin_keyword` post-response engine blocking `post_keyword`.
fn combined_guardrail(pre_keyword: &str, post_keyword: &str) -> GuardrailsConfig {
    GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![
            EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "pre_request".into(),
                action: Some("block".into()),
                keywords: vec![pre_keyword.into()],
                rules: vec![],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
            EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "post_response".into(),
                action: Some("block".into()),
                keywords: vec![post_keyword.into()],
                rules: vec![],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
        ],
    }
}

/// 13. With a combined pre + post pipeline: a request that contains the
///     pre-request blocked keyword is rejected with 403 before the upstream is
///     ever called.  The post-response engine never runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_combined_pre_blocks_before_upstream() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(combined_guardrail("BLOCKED_INPUT", "BLOCKED_OUTPUT"))
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    // Mount a mock that would return a clean response — it must never fire.
    openai::mock_chat_ok("gpt-4o", "A perfectly safe answer.")
        .mount(mock_server)
        .await;

    // Request contains the pre-request blocked keyword.
    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Please process this BLOCKED_INPUT for me."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        403,
        "request containing the pre-request blocked keyword must be rejected with 403"
    );

    // The upstream must have received zero requests — pre-request blocking
    // short-circuits the pipeline before forwarding.
    assert_received_n(mock_server, 0).await;
}

/// 14. With the same combined pre + post pipeline: a clean request passes the
///     pre-request engine, reaches the upstream, and is then blocked by the
///     post-response engine because the response contains the post keyword.
///     The upstream was called exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_combined_post_blocks_after_upstream() {
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(combined_guardrail("BLOCKED_INPUT", "BLOCKED_OUTPUT"))
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    // The upstream returns a response containing the post-response blocked keyword.
    openai::mock_chat_ok("gpt-4o", "Here is content that contains BLOCKED_OUTPUT.")
        .mount(mock_server)
        .await;

    // The request is clean — it does not contain BLOCKED_INPUT.
    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Please give me a helpful answer."}]
    });

    let resp = harness.client.chat_completions(body).await;
    // The post-response guardrail intercepts the toxic upstream response.
    assert_eq!(
        resp.status().as_u16(),
        403,
        "response containing the post-response blocked keyword must be rejected with 403"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["error"], "forbidden",
        "blocked response body must have error=forbidden"
    );

    // The upstream was called exactly once — the request passed the pre-request
    // engine and was forwarded to the provider before the response was blocked.
    assert_received_n(mock_server, 1).await;
}
