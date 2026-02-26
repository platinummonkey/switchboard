//! Integration tests for the HTTP callout guardrail engine.
//!
//! Each test spins up a wiremock server to simulate the external evaluator
//! service, then drives the engine directly against it.

use std::collections::HashMap;
use std::time::Duration;

use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use switchboard_server::config::guardrails::EngineConfig;
use switchboard_server::guardrails::builtin::engine_from_config;
use switchboard_server::guardrails::engine::{GuardrailAction, GuardrailEngine, GuardrailInput};
use switchboard_server::guardrails::http_callout::HttpCalloutEngine;
use switchboard_server::guardrails::pipeline::FailMode;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_input() -> GuardrailInput {
    GuardrailInput {
        content: "Hello, what is my SSN?".into(),
        messages: vec![],
        user: None,
        model: "claude-sonnet-4-20250514".into(),
        metadata: HashMap::new(),
    }
}

fn engine_for(
    server: &MockServer,
    fail_mode: FailMode,
    timeout: Duration,
    extra_headers: HashMap<String, String>,
) -> HttpCalloutEngine {
    HttpCalloutEngine::new(
        format!("{}/evaluate", server.uri()),
        timeout,
        fail_mode,
        extra_headers,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_http_callout_pass() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/evaluate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "pass",
            "rule": "",
            "reason": "",
            "confidence": 1.0
        })))
        .mount(&mock_server)
        .await;

    let engine = engine_for(
        &mock_server,
        FailMode::Open,
        Duration::from_secs(5),
        HashMap::new(),
    );
    let verdict = engine.evaluate_request(&make_input()).await.unwrap();

    assert!(
        verdict.action.is_pass(),
        "expected Pass action, got {:?}",
        verdict.action
    );
    assert_eq!(verdict.engine, "http_callout");
}

#[tokio::test]
async fn test_http_callout_block() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/evaluate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "block",
            "rule": "pii-detection",
            "reason": "Found SSN pattern",
            "confidence": 0.95
        })))
        .mount(&mock_server)
        .await;

    let engine = engine_for(
        &mock_server,
        FailMode::Open,
        Duration::from_secs(5),
        HashMap::new(),
    );
    let verdict = engine.evaluate_request(&make_input()).await.unwrap();

    assert!(
        verdict.action.is_blocking(),
        "expected Block action, got {:?}",
        verdict.action
    );
    assert_eq!(verdict.engine, "http_callout");
    assert_eq!(verdict.rule, "pii-detection");
    if let GuardrailAction::Block { message } = &verdict.action {
        assert_eq!(message, "Found SSN pattern");
    } else {
        panic!("expected Block variant");
    }
}

#[tokio::test]
async fn test_http_callout_modify() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/evaluate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "modify",
            "rule": "pii-redaction",
            "reason": "Redacted PII",
            "confidence": 0.99,
            "modified_content": "Hello, what is my [REDACTED]?"
        })))
        .mount(&mock_server)
        .await;

    let engine = engine_for(
        &mock_server,
        FailMode::Open,
        Duration::from_secs(5),
        HashMap::new(),
    );
    let verdict = engine.evaluate_request(&make_input()).await.unwrap();

    match &verdict.action {
        GuardrailAction::Modify { modified_content } => {
            assert_eq!(modified_content, "Hello, what is my [REDACTED]?");
        }
        other => panic!("expected Modify action, got {other:?}"),
    }
}

#[tokio::test]
async fn test_http_callout_audit_log() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/evaluate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "audit_log",
            "rule": "sensitive-topic",
            "reason": "Potentially sensitive",
            "confidence": 0.7
        })))
        .mount(&mock_server)
        .await;

    let engine = engine_for(
        &mock_server,
        FailMode::Open,
        Duration::from_secs(5),
        HashMap::new(),
    );
    let verdict = engine.evaluate_request(&make_input()).await.unwrap();

    assert!(
        matches!(verdict.action, GuardrailAction::AuditLog { .. }),
        "expected AuditLog action, got {:?}",
        verdict.action
    );
}

#[tokio::test]
async fn test_http_callout_fail_open_on_5xx() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/evaluate"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&mock_server)
        .await;

    let engine = engine_for(
        &mock_server,
        FailMode::Open,
        Duration::from_secs(5),
        HashMap::new(),
    );
    let verdict = engine.evaluate_request(&make_input()).await.unwrap();

    assert!(
        verdict.action.is_pass(),
        "fail-open: expected Pass on 5xx, got {:?}",
        verdict.action
    );
}

#[tokio::test]
async fn test_http_callout_fail_closed_on_5xx() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/evaluate"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&mock_server)
        .await;

    let engine = engine_for(
        &mock_server,
        FailMode::Closed,
        Duration::from_secs(5),
        HashMap::new(),
    );
    let verdict = engine.evaluate_request(&make_input()).await.unwrap();

    assert!(
        verdict.action.is_blocking(),
        "fail-closed: expected Block on 5xx, got {:?}",
        verdict.action
    );
}

#[tokio::test]
async fn test_http_callout_fail_open_on_timeout() {
    let mock_server = MockServer::start().await;

    // Delay the response well past the engine's 100ms timeout.
    Mock::given(method("POST"))
        .and(path("/evaluate"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(10))
                .set_body_json(serde_json::json!({
                    "action": "block",
                    "rule": "too_late",
                    "reason": "you shouldn't see this",
                    "confidence": 1.0
                })),
        )
        .mount(&mock_server)
        .await;

    // Very short timeout to trigger the timeout path.
    let engine = engine_for(
        &mock_server,
        FailMode::Open,
        Duration::from_millis(100),
        HashMap::new(),
    );
    let verdict = engine.evaluate_request(&make_input()).await.unwrap();

    assert!(
        verdict.action.is_pass(),
        "fail-open on timeout: expected Pass, got {:?}",
        verdict.action
    );
}

#[tokio::test]
async fn test_http_callout_custom_headers_forwarded() {
    let mock_server = MockServer::start().await;

    // The mock requires the Authorization header to be present.
    Mock::given(method("POST"))
        .and(path("/evaluate"))
        .and(header("Authorization", "Bearer secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "pass",
            "rule": "",
            "reason": "",
            "confidence": 1.0
        })))
        .mount(&mock_server)
        .await;

    let mut extra_headers = HashMap::new();
    extra_headers.insert("Authorization".into(), "Bearer secret".into());

    let engine = engine_for(
        &mock_server,
        FailMode::Open,
        Duration::from_secs(5),
        extra_headers,
    );
    let verdict = engine.evaluate_request(&make_input()).await.unwrap();

    // If the header was NOT forwarded, wiremock returns 404 and the engine
    // would fail-open (still Pass), but the mock would record no requests.
    // We verify the mock was matched by checking for a Pass from the real 200.
    assert!(
        verdict.action.is_pass(),
        "expected Pass with correct auth header, got {:?}",
        verdict.action
    );

    // Verify that the mock was actually hit (i.e., the header was sent).
    let received = mock_server.received_requests().await.unwrap();
    assert_eq!(
        received.len(),
        1,
        "expected exactly 1 request to the mock server"
    );
    assert_eq!(
        received[0]
            .headers
            .get("authorization")
            .map(|v| v.to_str().unwrap()),
        Some("Bearer secret")
    );
}

#[tokio::test]
async fn test_http_callout_request_body_contains_model() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/evaluate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "action": "pass",
            "rule": "",
            "reason": "",
            "confidence": 1.0
        })))
        .mount(&mock_server)
        .await;

    let engine = engine_for(
        &mock_server,
        FailMode::Open,
        Duration::from_secs(5),
        HashMap::new(),
    );
    let input = GuardrailInput {
        content: "test content".into(),
        messages: vec![],
        user: None,
        model: "claude-sonnet-4-20250514".into(),
        metadata: HashMap::new(),
    };

    let _verdict = engine.evaluate_request(&input).await.unwrap();

    // Inspect the raw request body sent to the mock server.
    let received = mock_server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);

    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(
        body.get("model").and_then(|v| v.as_str()),
        Some("claude-sonnet-4-20250514"),
        "request body must contain the 'model' field"
    );
}

#[tokio::test]
async fn test_engine_from_config_http_type() {
    // Verify that the factory correctly constructs an HttpCalloutEngine.
    let cfg = EngineConfig {
        engine_type: "http".into(),
        phase: "pre_request".into(),
        rules: vec![],
        action: None,
        keywords: vec![],
        max_input_tokens: None,
        max_output_tokens: None,
        endpoint: Some("https://guardrails.internal/evaluate".into()),
        timeout: Some("300ms".into()),
        tls: false,
        headers: Default::default(),
    };

    let engine = engine_from_config(&cfg).unwrap();
    assert_eq!(
        engine.name(),
        "http_callout",
        "engine_from_config with type='http' must return HttpCalloutEngine"
    );
}
