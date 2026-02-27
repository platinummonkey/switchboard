//! E2E scenario: gRPC callout guardrail integrated into the full proxy stack.
//!
//! Each test spins up a real in-process tonic gRPC mock server on a random
//! OS-assigned port, then builds a `TestHarness` configured with a "grpc"
//! guardrail engine pointing at that port. Requests travel through the full
//! TCP server → middleware stack → guardrail pipeline → upstream provider
//! path, unlike the unit tests in `grpc_guardrail_test.rs` which bypass the
//! proxy by calling the engine directly.
//!
//! Tests:
//! 1. Block verdict → 403 Forbidden
//! 2. Pass verdict → 200 OK (upstream response forwarded)
//! 3. Unavailable server (fail-open) → 200 OK (request allowed)

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;
use tokio::net::TcpListener;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use switchboard_server::config::guardrails::{EngineConfig, GuardrailsConfig};
use switchboard_server::guardrails::proto::guardrails_v1::{
    Action, EvaluateCompletionInput, EvaluateRequestInput, EvaluateResponse,
    guardrail_evaluator_server::{GuardrailEvaluator, GuardrailEvaluatorServer},
};

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai;

// ── Mock gRPC service ─────────────────────────────────────────────────────────

/// Minimal mock evaluator that returns a pre-configured `EvaluateResponse` for
/// every `EvaluateRequest` and `EvaluateCompletion` call.
struct MockEvaluatorService {
    response: Arc<Mutex<EvaluateResponse>>,
}

impl MockEvaluatorService {
    fn new(response: EvaluateResponse) -> Self {
        Self {
            response: Arc::new(Mutex::new(response)),
        }
    }
}

#[tonic::async_trait]
impl GuardrailEvaluator for MockEvaluatorService {
    async fn evaluate_request(
        &self,
        _request: Request<EvaluateRequestInput>,
    ) -> Result<Response<EvaluateResponse>, Status> {
        let resp = self.response.lock().unwrap().clone();
        Ok(Response::new(resp))
    }

    async fn evaluate_completion(
        &self,
        _request: Request<EvaluateCompletionInput>,
    ) -> Result<Response<EvaluateResponse>, Status> {
        let resp = self.response.lock().unwrap().clone();
        Ok(Response::new(resp))
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Bind a TCP listener on a random OS-assigned port and return the bound
/// listener together with the resolved local address.
async fn bind_random_port() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

/// Start the mock gRPC server and return its `SocketAddr`.
///
/// The server runs in a background tokio task and is kept alive for the
/// duration of the test via the spawned future.
async fn start_grpc_mock(service: MockEvaluatorService) -> SocketAddr {
    let (listener, addr) = bind_random_port().await;
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    tokio::spawn(async move {
        Server::builder()
            .add_service(GuardrailEvaluatorServer::new(service))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });

    // Give the server a brief moment to begin accepting connections.
    tokio::time::sleep(Duration::from_millis(20)).await;

    addr
}

/// Build a `GuardrailsConfig` with a single "grpc" pre-request engine that
/// targets `addr`.
///
/// The endpoint is passed as `"host:port"` (no scheme); `async_engine_from_config`
/// inside the server normalises it to `"http://host:port"` when `tls = false`.
fn grpc_guardrail_config(addr: SocketAddr) -> GuardrailsConfig {
    GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "2s".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "grpc".into(),
            phase: "pre_request".into(),
            endpoint: Some(format!("{}:{}", addr.ip(), addr.port())),
            timeout: Some("1s".into()),
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

// ── Tests ─────────────────────────────────────────────────────────────────────

/// 1. When the gRPC evaluator returns `Action::Block`, the proxy must respond
///    with `403 Forbidden` before forwarding the request to the upstream
///    provider.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_e2e_grpc_callout_blocks_on_block_verdict() {
    let block_response = EvaluateResponse {
        action: Action::Block as i32,
        rule: "pii_detected".into(),
        reason: "SSN found in request".into(),
        confidence: 0.99,
        modified_content: String::new(),
    };

    let grpc_addr = start_grpc_mock(MockEvaluatorService::new(block_response)).await;
    let cfg = grpc_guardrail_config(grpc_addr);

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(cfg)
        .build()
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "My SSN is 123-45-6789"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        403,
        "gRPC BLOCK verdict must cause the proxy to return 403 Forbidden"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["error"], "forbidden",
        "blocked response body must contain error=forbidden"
    );
}

/// 2. When the gRPC evaluator returns `Action::Pass`, the proxy must forward
///    the request to the upstream OpenAI provider and return its `200 OK`
///    response to the client.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_grpc_callout_passes_on_allow_verdict() {
    let pass_response = EvaluateResponse {
        action: Action::Pass as i32,
        rule: String::new(),
        reason: String::new(),
        confidence: 1.0,
        modified_content: String::new(),
    };

    let grpc_addr = start_grpc_mock(MockEvaluatorService::new(pass_response)).await;
    let cfg = grpc_guardrail_config(grpc_addr);

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_guardrails(cfg)
        .build()
        .await;

    // Mount the upstream mock so the request has somewhere to go.
    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    openai::mock_chat_ok("gpt-4o", "The capital of France is Paris.")
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
        "gRPC PASS verdict must allow the request through with 200 OK"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["choices"][0]["message"]["content"], "The capital of France is Paris.",
        "response content must match the upstream mock"
    );
}

/// 3. When the configured gRPC endpoint is unreachable (no server listening at
///    that port) and `fail_mode` is `"open"`, the guardrail must fail-open:
///    the request is allowed through and the upstream returns `200 OK`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_grpc_callout_fail_open_on_unavailable_server() {
    // Allocate a port but immediately drop the listener so nothing is listening.
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        listener.local_addr().unwrap().port()
        // listener dropped here — port is free but nothing is listening
    };
    let unreachable_addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

    // Build a guardrail config pointing at the dead port with fail_mode=open.
    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "2s".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "grpc".into(),
            phase: "pre_request".into(),
            endpoint: Some(format!(
                "{}:{}",
                unreachable_addr.ip(),
                unreachable_addr.port()
            )),
            // Short timeout so the test doesn't stall waiting for a connection.
            timeout: Some("200ms".into()),
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
        .with_guardrails(cfg)
        .build()
        .await;

    let mock_server = harness
        .mocks
        .openai
        .as_ref()
        .expect("openai mock must be present");
    openai::mock_chat_ok("gpt-4o", "Fail-open allowed this through.")
        .mount(mock_server)
        .await;

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello, is anyone there?"}]
    });

    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "fail-open mode must allow requests when the gRPC evaluator is unreachable (expected 200 OK)"
    );

    let json: serde_json::Value = resp.json().await.expect("response must be valid JSON");
    assert_eq!(
        json["choices"][0]["message"]["content"], "Fail-open allowed this through.",
        "upstream response must be forwarded unchanged after fail-open"
    );
}
