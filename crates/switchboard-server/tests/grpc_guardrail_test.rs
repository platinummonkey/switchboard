//! Integration tests for the gRPC guardrail callout engine.
//!
//! Each test spins up an in-process mock gRPC server, connects a
//! `GrpcCalloutEngine` to it, and verifies that the returned
//! `GuardrailVerdict` matches the expected `GuardrailAction`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::TcpListener;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use switchboard_server::guardrails::GrpcCalloutEngine;
use switchboard_server::guardrails::engine::{
    AuditSeverity, GuardrailAction, GuardrailEngine, GuardrailInput,
};
use switchboard_server::guardrails::pipeline::FailMode;
use switchboard_server::guardrails::proto::guardrails_v1::{
    Action, EvaluateCompletionInput, EvaluateRequestInput, EvaluateResponse,
    guardrail_evaluator_server::{GuardrailEvaluator, GuardrailEvaluatorServer},
};

// ── Mock gRPC service ─────────────────────────────────────────────────────────

/// A test gRPC server that returns configurable responses.
struct MockEvaluatorService {
    /// Mutex-guarded response returned for every call.
    response: Arc<Mutex<EvaluateResponse>>,
    /// If Some, sleep this long before responding (used to test timeouts).
    sleep_duration: Option<Duration>,
    /// If true, return an internal status error instead of the configured response.
    return_error: bool,
}

impl MockEvaluatorService {
    fn new(response: EvaluateResponse) -> Self {
        Self {
            response: Arc::new(Mutex::new(response)),
            sleep_duration: None,
            return_error: false,
        }
    }

    fn with_sleep(mut self, d: Duration) -> Self {
        self.sleep_duration = Some(d);
        self
    }

    fn with_error(mut self) -> Self {
        self.return_error = true;
        self
    }
}

#[tonic::async_trait]
impl GuardrailEvaluator for MockEvaluatorService {
    async fn evaluate_request(
        &self,
        _request: Request<EvaluateRequestInput>,
    ) -> Result<Response<EvaluateResponse>, Status> {
        if let Some(sleep) = self.sleep_duration {
            tokio::time::sleep(sleep).await;
        }
        if self.return_error {
            return Err(Status::internal("intentional test error"));
        }
        let resp = self.response.lock().unwrap().clone();
        Ok(Response::new(resp))
    }

    async fn evaluate_completion(
        &self,
        _request: Request<EvaluateCompletionInput>,
    ) -> Result<Response<EvaluateResponse>, Status> {
        if let Some(sleep) = self.sleep_duration {
            tokio::time::sleep(sleep).await;
        }
        if self.return_error {
            return Err(Status::internal("intentional test error"));
        }
        let resp = self.response.lock().unwrap().clone();
        Ok(Response::new(resp))
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Bind a TCP listener on a random OS-assigned port and return the address.
async fn bind_random_port() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

/// Start the mock server and return its address.
async fn start_mock_server(service: MockEvaluatorService) -> SocketAddr {
    let (listener, addr) = bind_random_port().await;
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    tokio::spawn(async move {
        Server::builder()
            .add_service(GuardrailEvaluatorServer::new(service))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });

    // Give the server a moment to start listening.
    tokio::time::sleep(Duration::from_millis(10)).await;

    addr
}

/// Build a minimal `GuardrailInput` for testing.
fn make_input(content: &str) -> GuardrailInput {
    GuardrailInput {
        content: content.into(),
        messages: vec![],
        user: None,
        model: "test-model".into(),
        metadata: HashMap::new(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_grpc_callout_pass() {
    let response = EvaluateResponse {
        action: Action::Pass as i32,
        rule: String::new(),
        reason: String::new(),
        confidence: 1.0,
        modified_content: String::new(),
    };

    let addr = start_mock_server(MockEvaluatorService::new(response)).await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(&endpoint, Duration::from_secs(5), FailMode::Open)
        .await
        .expect("connect should succeed");

    let verdict = engine
        .evaluate_request(&make_input("hello world"))
        .await
        .expect("evaluate_request should succeed");

    assert!(
        verdict.action.is_pass(),
        "PASS action should produce a passing verdict; got {:?}",
        verdict.action
    );
    assert_eq!(verdict.engine, "grpc_callout");
}

#[tokio::test]
async fn test_grpc_callout_block() {
    let response = EvaluateResponse {
        action: Action::Block as i32,
        rule: "pii_detected".into(),
        reason: "SSN found in request".into(),
        confidence: 0.99,
        modified_content: String::new(),
    };

    let addr = start_mock_server(MockEvaluatorService::new(response)).await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(&endpoint, Duration::from_secs(5), FailMode::Open)
        .await
        .unwrap();

    let verdict = engine
        .evaluate_request(&make_input("My SSN is 123-45-6789"))
        .await
        .unwrap();

    assert!(
        verdict.action.is_blocking(),
        "BLOCK action should produce a blocking verdict; got {:?}",
        verdict.action
    );
    assert_eq!(verdict.rule, "pii_detected");

    match verdict.action {
        GuardrailAction::Block { message } => {
            assert_eq!(message, "SSN found in request");
        }
        other => panic!("expected Block, got {other:?}"),
    }
}

#[tokio::test]
async fn test_grpc_callout_modify() {
    let response = EvaluateResponse {
        action: Action::Modify as i32,
        rule: "redact_pii".into(),
        reason: "PII replaced".into(),
        confidence: 0.95,
        modified_content: "My SSN is [REDACTED]".into(),
    };

    let addr = start_mock_server(MockEvaluatorService::new(response)).await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(&endpoint, Duration::from_secs(5), FailMode::Open)
        .await
        .unwrap();

    let verdict = engine
        .evaluate_request(&make_input("My SSN is 123-45-6789"))
        .await
        .unwrap();

    match verdict.action {
        GuardrailAction::Modify { modified_content } => {
            assert_eq!(modified_content, "My SSN is [REDACTED]");
        }
        other => panic!("expected Modify, got {other:?}"),
    }

    assert_eq!(verdict.rule, "redact_pii");
}

#[tokio::test]
async fn test_grpc_callout_audit_log() {
    let response = EvaluateResponse {
        action: Action::AuditLog as i32,
        rule: "suspicious_query".into(),
        reason: "Unusual request pattern".into(),
        confidence: 0.75,
        modified_content: String::new(),
    };

    let addr = start_mock_server(MockEvaluatorService::new(response)).await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(&endpoint, Duration::from_secs(5), FailMode::Open)
        .await
        .unwrap();

    let verdict = engine
        .evaluate_request(&make_input("unusual query here"))
        .await
        .unwrap();

    match verdict.action {
        GuardrailAction::AuditLog { severity } => {
            assert_eq!(severity, AuditSeverity::Warning);
        }
        other => panic!("expected AuditLog, got {other:?}"),
    }

    assert!((verdict.confidence - 0.75).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_grpc_callout_fail_open_on_server_error() {
    let addr =
        start_mock_server(MockEvaluatorService::new(EvaluateResponse::default()).with_error())
            .await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(&endpoint, Duration::from_secs(5), FailMode::Open)
        .await
        .unwrap();

    let verdict = engine.evaluate_request(&make_input("test")).await.unwrap();

    assert!(
        verdict.action.is_pass(),
        "fail-open should return Pass on server error; got {:?}",
        verdict.action
    );
}

#[tokio::test]
async fn test_grpc_callout_fail_closed_on_server_error() {
    let addr =
        start_mock_server(MockEvaluatorService::new(EvaluateResponse::default()).with_error())
            .await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(&endpoint, Duration::from_secs(5), FailMode::Closed)
        .await
        .unwrap();

    let verdict = engine.evaluate_request(&make_input("test")).await.unwrap();

    assert!(
        verdict.action.is_blocking(),
        "fail-closed should return Block on server error; got {:?}",
        verdict.action
    );
    assert_eq!(verdict.rule, "engine_error");
}

#[tokio::test]
async fn test_grpc_callout_timeout_fail_open() {
    // Server sleeps 300 ms; engine timeout is 50 ms → should time out.
    let addr = start_mock_server(
        MockEvaluatorService::new(EvaluateResponse::default())
            .with_sleep(Duration::from_millis(300)),
    )
    .await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(
        &endpoint,
        Duration::from_millis(50), // tight timeout
        FailMode::Open,
    )
    .await
    .unwrap();

    let verdict = engine.evaluate_request(&make_input("test")).await.unwrap();

    assert!(
        verdict.action.is_pass(),
        "fail-open timeout should return Pass; got {:?}",
        verdict.action
    );
}

#[tokio::test]
async fn test_grpc_callout_evaluate_response_pass() {
    let response = EvaluateResponse {
        action: Action::Pass as i32,
        rule: String::new(),
        reason: String::new(),
        confidence: 1.0,
        modified_content: String::new(),
    };

    let addr = start_mock_server(MockEvaluatorService::new(response)).await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(&endpoint, Duration::from_secs(5), FailMode::Open)
        .await
        .unwrap();

    let verdict = engine
        .evaluate_response(&make_input("This is an LLM completion"))
        .await
        .unwrap();

    assert!(
        verdict.action.is_pass(),
        "evaluate_response PASS should return passing verdict"
    );
}

#[tokio::test]
async fn test_grpc_callout_evaluate_response_block() {
    let response = EvaluateResponse {
        action: Action::Block as i32,
        rule: "harmful_output".into(),
        reason: "Response contains harmful content".into(),
        confidence: 0.98,
        modified_content: String::new(),
    };

    let addr = start_mock_server(MockEvaluatorService::new(response)).await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(&endpoint, Duration::from_secs(5), FailMode::Closed)
        .await
        .unwrap();

    let verdict = engine
        .evaluate_response(&make_input("harmful output text"))
        .await
        .unwrap();

    assert!(
        verdict.action.is_blocking(),
        "evaluate_response BLOCK should return blocking verdict"
    );
    assert_eq!(verdict.rule, "harmful_output");
}

#[tokio::test]
async fn test_grpc_callout_timeout_fail_closed() {
    // Server sleeps 300 ms; engine timeout is 50 ms → should time out, and
    // fail-closed returns Block.
    let addr = start_mock_server(
        MockEvaluatorService::new(EvaluateResponse::default())
            .with_sleep(Duration::from_millis(300)),
    )
    .await;
    let endpoint = format!("http://{addr}");

    let engine = GrpcCalloutEngine::connect(&endpoint, Duration::from_millis(50), FailMode::Closed)
        .await
        .unwrap();

    let verdict = engine.evaluate_request(&make_input("test")).await.unwrap();

    assert!(
        verdict.action.is_blocking(),
        "fail-closed timeout should return Block; got {:?}",
        verdict.action
    );
}
