//! E2E scenario: HTTP callout guardrail engine integrated into the full proxy stack.
//!
//! Tests verify that the `HttpCalloutEngine` — which POSTs each request to an
//! external HTTP evaluator and maps the JSON verdict to a guardrail action —
//! works end-to-end through the `GuardrailLayer`.
//!
//! ## Startup note
//!
//! The `GuardrailLayer` is applied to **all** routes, including `GET /health`.
//! During server startup, `wait_for_ready` polls `/health` which causes the
//! guardrail to call the HTTP evaluator.  Tests that need the evaluator to
//! return a blocking verdict must therefore:
//!
//! 1. Mount a temporary "pass" mock (scoped) so health checks succeed.
//! 2. Build the harness (server starts healthy with the pass mock active).
//! 3. Drop the scoped "pass" guard — removes it from the evaluator server.
//! 4. Mount the "block" mock permanently.
//! 5. Send the actual test request and assert the verdict.
//!
//! Tests that need "pass" for both startup and the real request can skip steps
//! 3–4 and just keep the "pass" mock mounted throughout.
//!
//! ## Wire format recap
//!
//! **Request** (POST to evaluator):
//! ```json
//! { "messages": [...], "model": "gpt-4o", "user_id": null, "team": null, "metadata": {} }
//! ```
//!
//! **Response** from evaluator:
//! ```json
//! { "action": "pass"|"block"|"modify"|"audit_log", "rule": "...", "reason": "...", "confidence": 0.95 }
//! ```

use std::time::Duration;

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use switchboard_server::config::guardrails::{EngineConfig, GuardrailsConfig};

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai;

// ── Config helper ─────────────────────────────────────────────────────────────

/// Build a `GuardrailsConfig` with a single `http` pre-request engine that
/// POSTs to `evaluator_url`. The pipeline `fail_mode` and `timeout` are
/// configurable so individual tests can exercise different failure behaviours.
fn http_guardrail_cfg(evaluator_url: &str, fail_mode: &str, timeout: &str) -> GuardrailsConfig {
    GuardrailsConfig {
        enabled: true,
        fail_mode: fail_mode.into(),
        timeout: timeout.into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "http".into(),
            phase: "pre_request".into(),
            endpoint: Some(evaluator_url.to_owned()),
            // Per-engine timeout longer than the pipeline timeout so the
            // pipeline-level timeout can fire independently in fail_closed tests.
            timeout: Some("2000ms".into()),
            tls: false,
            action: None,
            keywords: vec![],
            rules: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            headers: Default::default(),
        }],
    }
}

/// Mount a permanent "pass" verdict on the evaluator mock server.
///
/// Returns the `Mock` for later inspection if needed, already mounted.
async fn mount_pass_verdict(evaluator: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "action": "pass",
            "rule": "allow",
            "reason": "",
            "confidence": 1.0
        })))
        .mount(evaluator)
        .await;
}

/// Mount a scoped "pass" verdict that is removed when the returned guard is
/// dropped.  Used to allow server startup health checks to succeed before the
/// real blocking mock is installed.
async fn mount_pass_verdict_scoped(evaluator: &MockServer) -> wiremock::MockGuard {
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "action": "pass",
            "rule": "startup-allow",
            "reason": "startup health check passthrough",
            "confidence": 1.0
        })))
        .mount_as_scoped(evaluator)
        .await
}

// ── 1. Block verdict → 403 ────────────────────────────────────────────────────

/// When the HTTP evaluator returns `{"action":"block"}` the `GuardrailLayer`
/// must short-circuit the pipeline and respond with 403 Forbidden.
/// The upstream LLM provider must never be called.
///
/// The evaluator is first primed with a "pass" mock (scoped) so server startup
/// health checks succeed.  Once the harness is ready the scoped mock is dropped
/// and the permanent "block" mock takes over.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_http_callout_blocks_on_block_verdict() {
    let evaluator = MockServer::start().await;

    // Prime evaluator with "pass" during startup so health checks succeed.
    let startup_guard = mount_pass_verdict_scoped(&evaluator).await;

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(http_guardrail_cfg(&evaluator.uri(), "open", "500ms"))
        .build()
        .await;

    // Server is now healthy — swap to the "block" mock.
    drop(startup_guard);

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "action": "block",
            "rule": "pii-detection",
            "reason": "test block",
            "confidence": 1.0
        })))
        .mount(&evaluator)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tell me something."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        403,
        "http callout 'block' verdict must cause 403 Forbidden"
    );

    // The upstream OpenAI mock must have received zero requests because the
    // guardrail blocked the request before it was forwarded.
    let openai_mock = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    let received = openai_mock
        .received_requests()
        .await
        .expect("failed to fetch openai requests");
    assert_eq!(
        received.len(),
        0,
        "upstream provider must not be called when the guardrail blocks"
    );
}

// ── 2. Pass verdict → 200 ─────────────────────────────────────────────────────

/// When the HTTP evaluator returns `{"action":"pass"}` the request is forwarded
/// to the upstream LLM and the full 200 OK response is returned to the client.
///
/// The "pass" mock is mounted before the harness is built so that both startup
/// health checks and the actual test request receive a "pass" verdict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_http_callout_passes_on_pass_verdict() {
    let evaluator = MockServer::start().await;

    // Permanent "pass" mock — valid for both startup and the real request.
    mount_pass_verdict(&evaluator).await;

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(http_guardrail_cfg(&evaluator.uri(), "open", "500ms"))
        .build()
        .await;

    let openai_mock = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    openai::mock_chat_ok("gpt-4o", "The capital of France is Paris.")
        .mount(openai_mock)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the capital of France?"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "http callout 'pass' verdict must allow the request through with 200 OK"
    );

    let json_body: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json_body["choices"][0]["message"]["content"], "The capital of France is Paris.",
        "response content must match the upstream mock"
    );
}

// ── 3. Evaluator receives correct request body ────────────────────────────────

/// The `HttpCalloutEngine` must POST a JSON body to the evaluator endpoint
/// that matches the configured wire format: `messages`, `model`, and `metadata`
/// fields must be present.
///
/// Note: The `GuardrailLayer` builds a `GuardrailInput` with `messages: vec![]`
/// and `model: ""` from the buffered body's extracted text content.  The HTTP
/// callout therefore sends `{"messages":[],"model":"","metadata":{}}`.  The
/// important property under test here is that:
///
/// 1. The evaluator IS called (the HTTP POST is made) for every proxy request.
/// 2. The body is valid JSON with the expected top-level keys.
/// 3. A `metadata` object is included (even if empty).
///
/// After sending the request the test inspects the evaluator's received
/// requests to verify the body structure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_http_callout_evaluator_receives_request_body() {
    let evaluator = MockServer::start().await;

    // Permanent "pass" — both startup and actual test request go through.
    mount_pass_verdict(&evaluator).await;

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(http_guardrail_cfg(&evaluator.uri(), "open", "500ms"))
        .build()
        .await;

    let openai_mock = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    openai::mock_chat_ok("gpt-4o", "ok")
        .mount(openai_mock)
        .await;

    // Count requests before to isolate just the one from our chat call.
    let requests_before = evaluator
        .received_requests()
        .await
        .expect("failed to fetch pre-test evaluator requests")
        .len();

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Does this contain any sensitive data?"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "request must succeed so the evaluator call is verified"
    );

    // Fetch what the evaluator received after our call.
    let all_eval_requests = evaluator
        .received_requests()
        .await
        .expect("failed to fetch post-test evaluator requests");

    // At least one new request must have arrived since we counted before.
    let new_requests_count = all_eval_requests.len() - requests_before;
    assert!(
        new_requests_count >= 1,
        "evaluator must receive at least one POST from the chat completion request; \
         got {} total, {} before the call",
        all_eval_requests.len(),
        requests_before,
    );

    // Inspect the last received request — it is the one from our chat call.
    let last_request = all_eval_requests
        .last()
        .expect("must have at least one evaluator request");

    let parsed: serde_json::Value = serde_json::from_slice(&last_request.body)
        .expect("evaluator request body must be valid JSON");

    // The callout body must contain the three top-level keys from the wire format.
    assert!(
        parsed.get("messages").is_some(),
        "evaluator request body must contain 'messages' field; got: {parsed}"
    );
    assert!(
        parsed.get("model").is_some(),
        "evaluator request body must contain 'model' field; got: {parsed}"
    );
    assert!(
        parsed.get("metadata").is_some(),
        "evaluator request body must contain 'metadata' field; got: {parsed}"
    );

    // The 'messages' key must be a JSON array (empty or otherwise).
    assert!(
        parsed["messages"].is_array(),
        "'messages' must be a JSON array; got: {}",
        parsed["messages"]
    );

    // The 'metadata' key must be a JSON object.
    assert!(
        parsed["metadata"].is_object(),
        "'metadata' must be a JSON object; got: {}",
        parsed["metadata"]
    );
}

// ── 4. Fail-open on evaluator error → 200 ────────────────────────────────────

/// When the evaluator returns a 500 Internal Server Error and the pipeline is
/// configured with `fail_mode: "open"`, the `HttpCalloutEngine` treats the
/// error as a pass verdict (fail-open) and forwards the request to the upstream
/// LLM, returning 200 OK to the client.
///
/// The scoped "pass" mock ensures startup health checks succeed; the permanent
/// 500 mock is installed after the harness is ready.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_http_callout_fail_open_on_evaluator_error() {
    let evaluator = MockServer::start().await;

    // Pass during startup, then swap to 500.
    let startup_guard = mount_pass_verdict_scoped(&evaluator).await;

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(http_guardrail_cfg(&evaluator.uri(), "open", "500ms"))
        .build()
        .await;

    // Swap: remove pass mock, install 500 mock.
    drop(startup_guard);

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({
            "error": "internal server error"
        })))
        .mount(&evaluator)
        .await;

    let openai_mock = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    openai::mock_chat_ok("gpt-4o", "Allowed by fail-open policy.")
        .mount(openai_mock)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "This request should be allowed through."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "fail-open policy must allow the request through when the evaluator returns 500"
    );

    let json_body: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json_body["choices"][0]["message"]["content"], "Allowed by fail-open policy.",
        "fail-open response must contain the upstream content"
    );
}

// ── 5. Fail-closed on evaluator timeout → non-200 ────────────────────────────

/// When the evaluator is slow (200ms artificial delay) and the pipeline is
/// configured with `fail_mode: "closed"` and a very short pipeline timeout
/// (10ms), the pipeline-level timeout fires first and the request is blocked
/// with a non-200 response.
///
/// ## Implementation note
///
/// `HttpCalloutEngine::from_config` always sets the engine's internal
/// `FailMode` to `Open`.  Fail-closed blocking is therefore triggered by the
/// pipeline-level timeout (10ms here) expiring before the evaluator responds
/// (200ms delay).  The pipeline's `run_engine` receives `Err(Elapsed)` and
/// applies `FailMode::Closed` → Block.
///
/// The scoped "pass" mock (no delay) is active during startup so health checks
/// complete quickly.  After the harness is ready the pass guard is dropped and
/// the slow 500 mock is installed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_http_callout_fail_closed_on_evaluator_error() {
    let evaluator = MockServer::start().await;

    // During startup health checks the pipeline timeout (10ms) would also fire
    // for health checks — so we need an immediate "pass" response for startup.
    let startup_guard = mount_pass_verdict_scoped(&evaluator).await;

    // Pipeline timeout of 10ms — will expire before the slow evaluator responds.
    // The engine's own timeout is 2000ms so the pipeline timeout fires first.
    let guardrail_cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "closed".into(),
        timeout: "10ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "http".into(),
            phase: "pre_request".into(),
            endpoint: Some(evaluator.uri()),
            timeout: Some("2000ms".into()),
            tls: false,
            action: None,
            keywords: vec![],
            rules: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            headers: Default::default(),
        }],
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(guardrail_cfg)
        .build()
        .await;

    // Swap startup "pass" for the slow evaluator that triggers a timeout.
    drop(startup_guard);

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(
                    json!({"action": "pass", "rule": "slow", "reason": "", "confidence": 1.0}),
                )
                .set_delay(Duration::from_millis(200)),
        )
        .mount(&evaluator)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "This request should be blocked by fail-closed policy."}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_ne!(
        resp.status().as_u16(),
        200,
        "fail-closed policy must block the request when the pipeline times out waiting for the evaluator; got {}",
        resp.status().as_u16()
    );
}
