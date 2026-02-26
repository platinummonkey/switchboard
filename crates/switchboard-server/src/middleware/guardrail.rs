//! Guardrail Tower middleware layer.
//!
//! [`GuardrailLayer`] wraps the proxy service with guardrail evaluation:
//!
//! - **Pre-request**: buffers the incoming body, runs the guardrail pipeline's
//!   pre-request engines, and either blocks the request (returns 403) or
//!   forwards it with the (potentially modified) body.
//!
//! - **Post-response (non-streaming)**: buffers the response body, runs the
//!   post-response engines, and either replaces the response with a 403 block
//!   message or returns the original / modified response.
//!
//! - **Post-response (streaming)**: uses `async_audit` mode — the response
//!   stream is forwarded immediately and evaluation is fired asynchronously in
//!   the background via [`tokio::spawn`].

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::{Body, to_bytes};
use http::{Request, Response, StatusCode};
use tower::{Layer, Service};

use crate::config::guardrails::GuardrailsConfig;
use crate::error::ServerError;
use crate::guardrails::engine::{GuardrailAction, GuardrailInput};
use crate::guardrails::pipeline::GuardrailPipeline;

// ── GuardrailLayer ────────────────────────────────────────────────────────────

/// Tower [`Layer`] that wraps services with guardrail evaluation.
#[derive(Clone)]
pub struct GuardrailLayer {
    pipeline: Arc<GuardrailPipeline>,
}

impl std::fmt::Debug for GuardrailLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailLayer")
            .field("pipeline", &self.pipeline)
            .finish()
    }
}

impl GuardrailLayer {
    /// Build from the server's guardrails config.
    ///
    /// Returns `Ok(None)` if guardrails are disabled in config.
    /// Returns `Ok(Some(layer))` if guardrails are enabled and the pipeline
    /// was successfully constructed.
    pub fn from_config(config: &GuardrailsConfig) -> Result<Option<Self>, ServerError> {
        if !config.enabled {
            return Ok(None);
        }
        let pipeline = GuardrailPipeline::from_config(config)?;
        Ok(Some(Self::new(Arc::new(pipeline))))
    }

    /// Build a layer from an already-constructed pipeline.
    pub fn new(pipeline: Arc<GuardrailPipeline>) -> Self {
        Self { pipeline }
    }
}

impl<S> Layer<S> for GuardrailLayer {
    type Service = GuardrailService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GuardrailService {
            inner,
            pipeline: Arc::clone(&self.pipeline),
        }
    }
}

// ── GuardrailService ──────────────────────────────────────────────────────────

/// Tower [`Service`] produced by [`GuardrailLayer`].
#[derive(Clone)]
pub struct GuardrailService<S> {
    inner: S,
    pipeline: Arc<GuardrailPipeline>,
}

impl<S> std::fmt::Debug for GuardrailService<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailService")
            .field("pipeline", &self.pipeline)
            .finish_non_exhaustive()
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

impl<S> Service<Request<Body>> for GuardrailService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let pipeline = Arc::clone(&self.pipeline);
        let mut inner = self.inner.clone();

        Box::pin(async move {
            // ── Step 1: buffer the request body ──────────────────────────────
            let (parts, body) = req.into_parts();
            let body_bytes = match to_bytes(body, usize::MAX).await {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to buffer request body for guardrail check");
                    return Ok(error_response(
                        StatusCode::BAD_REQUEST,
                        "failed to read request body",
                    ));
                }
            };

            // ── Step 2: extract text content for evaluation ───────────────────
            let content = extract_content_from_body(&body_bytes);

            // ── Step 3: build GuardrailInput and run pre-request evaluation ───
            let input = GuardrailInput {
                content,
                messages: vec![],
                user: None,
                model: String::new(),
                metadata: HashMap::new(),
            };

            let verdict = pipeline.evaluate_request(&input).await;

            // ── Step 4: handle verdict ────────────────────────────────────────
            match &verdict.action {
                GuardrailAction::Block { message } => {
                    tracing::warn!(
                        engine = %verdict.engine,
                        rule = %verdict.rule,
                        message = %message,
                        "guardrail blocked request"
                    );
                    return Ok(blocked_response(message));
                }
                GuardrailAction::Modify { modified_content } => {
                    tracing::debug!(
                        engine = %verdict.engine,
                        rule = %verdict.rule,
                        "guardrail modified request body"
                    );
                    // Re-serialize with modified content substituted into the body JSON.
                    let new_body_bytes = rewrite_body_content(&body_bytes, modified_content);
                    let new_req = Request::from_parts(parts, Body::from(new_body_bytes));
                    let response = inner.call(new_req).await?;
                    return Ok(apply_post_response_guardrails(response, pipeline).await);
                }
                GuardrailAction::AuditLog { severity } => {
                    tracing::info!(
                        engine = %verdict.engine,
                        rule = %verdict.rule,
                        severity = %severity,
                        "guardrail audit log (pre-request)"
                    );
                }
                GuardrailAction::Pass => {}
            }

            // ── Step 5: forward the request with the original (buffered) body ─
            let is_streaming = is_streaming_request(&body_bytes);
            let new_req = Request::from_parts(parts, Body::from(body_bytes));
            let response = inner.call(new_req).await?;

            // ── Step 6: post-response guardrail evaluation ────────────────────
            if is_streaming || is_streaming_response(&response) {
                // Async audit mode: spawn background task, let stream through.
                let response_pipeline = Arc::clone(&pipeline);
                tokio::spawn(async move {
                    tracing::debug!("streaming response: async_audit guardrail evaluation");
                    let audit_input = GuardrailInput {
                        content: "[streaming response — async audit]".into(),
                        messages: vec![],
                        user: None,
                        model: String::new(),
                        metadata: HashMap::new(),
                    };
                    let verdict = response_pipeline.evaluate_response(&audit_input).await;
                    if !verdict.action.is_pass() {
                        tracing::warn!(
                            engine = %verdict.engine,
                            rule = %verdict.rule,
                            action = %verdict.action,
                            "streaming guardrail post-response audit"
                        );
                    }
                });
                Ok(response)
            } else {
                Ok(apply_post_response_guardrails(response, pipeline).await)
            }
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Apply post-response guardrail evaluation to a non-streaming response.
async fn apply_post_response_guardrails(
    response: Response<Body>,
    pipeline: Arc<GuardrailPipeline>,
) -> Response<Body> {
    // Only evaluate non-streaming responses.
    if is_streaming_response(&response) {
        return response;
    }

    let (resp_parts, resp_body) = response.into_parts();
    let resp_bytes = match to_bytes(resp_body, usize::MAX).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "failed to buffer response body for post-guardrail check");
            // Return a minimal error response.
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "failed to read response");
        }
    };

    let content = extract_content_from_body(&resp_bytes);
    let input = GuardrailInput {
        content,
        messages: vec![],
        user: None,
        model: String::new(),
        metadata: HashMap::new(),
    };

    let verdict = pipeline.evaluate_response(&input).await;
    match &verdict.action {
        GuardrailAction::Block { message } => {
            tracing::warn!(
                engine = %verdict.engine,
                rule = %verdict.rule,
                message = %message,
                "guardrail blocked response"
            );
            blocked_response(message)
        }
        GuardrailAction::Modify { modified_content } => {
            tracing::debug!(
                engine = %verdict.engine,
                rule = %verdict.rule,
                "guardrail modified response body"
            );
            let new_bytes = rewrite_body_content(&resp_bytes, modified_content);
            Response::from_parts(resp_parts, Body::from(new_bytes))
        }
        GuardrailAction::AuditLog { severity } => {
            tracing::info!(
                engine = %verdict.engine,
                rule = %verdict.rule,
                severity = %severity,
                "guardrail audit log (post-response)"
            );
            Response::from_parts(resp_parts, Body::from(resp_bytes))
        }
        GuardrailAction::Pass => Response::from_parts(resp_parts, Body::from(resp_bytes)),
    }
}

/// Check whether the request body JSON has `"stream": true`.
pub fn is_streaming_request(body_bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(body_bytes)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false)
}

/// Check whether the response is a streaming SSE response.
fn is_streaming_response(response: &Response<Body>) -> bool {
    response
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("text/event-stream"))
        .unwrap_or(false)
}

/// Extract text content from the request/response body JSON.
///
/// For JSON objects with a `messages` array, concatenates all `content` text
/// fields. Falls back to the raw UTF-8 body if not valid JSON.
fn extract_content_from_body(body_bytes: &[u8]) -> String {
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(body_bytes) {
        // Try to extract messages[*].content
        if let Some(messages) = json.get("messages").and_then(|m| m.as_array()) {
            let parts: Vec<String> = messages
                .iter()
                .filter_map(|msg| {
                    let content = msg.get("content")?;
                    // content can be a string or an array of parts
                    if let Some(s) = content.as_str() {
                        Some(s.to_owned())
                    } else if let Some(arr) = content.as_array() {
                        let texts: Vec<&str> = arr
                            .iter()
                            .filter_map(|part| {
                                if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                                    part.get("text").and_then(|t| t.as_str())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        if texts.is_empty() {
                            None
                        } else {
                            Some(texts.join("\n"))
                        }
                    } else {
                        None
                    }
                })
                .collect();
            if !parts.is_empty() {
                return parts.join("\n");
            }
        }

        // Fallback: extract top-level "content" field if present.
        if let Some(s) = json.get("content").and_then(|c| c.as_str()) {
            return s.to_owned();
        }

        // Render the entire JSON as text.
        return json.to_string();
    }

    // Not JSON: use raw bytes as UTF-8.
    String::from_utf8_lossy(body_bytes).into_owned()
}

/// Replace the body JSON's `messages[*].content` with `modified_content`, or
/// fall back to using `modified_content` directly as the new body.
fn rewrite_body_content(original: &[u8], modified_content: &str) -> bytes::Bytes {
    if let Ok(mut json) = serde_json::from_slice::<serde_json::Value>(original) {
        if let Some(messages) = json.get_mut("messages").and_then(|m| m.as_array_mut()) {
            // Replace the last user message content with the modified text.
            for msg in messages.iter_mut().rev() {
                if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                    msg["content"] = serde_json::Value::String(modified_content.to_owned());
                    break;
                }
            }
        }
        let serialized = serde_json::to_vec(&json).unwrap_or_else(|_| original.to_vec());
        return bytes::Bytes::from(serialized);
    }
    bytes::Bytes::from(modified_content.to_owned())
}

/// Build a JSON 403 Forbidden response.
fn blocked_response(message: &str) -> Response<Body> {
    let body = serde_json::json!({
        "error": "forbidden",
        "message": message,
    })
    .to_string();
    Response::builder()
        .status(StatusCode::FORBIDDEN)
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("static response is always valid")
}

/// Build an error response with the given status and message.
fn error_response(status: StatusCode, message: &str) -> Response<Body> {
    let body = serde_json::json!({
        "error": status.canonical_reason().unwrap_or("error"),
        "message": message,
    })
    .to_string();
    Response::builder()
        .status(status)
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("static response is always valid")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;

    use axum::body::Body;
    use http::{Request, Response, StatusCode};
    use tower::{Layer, Service, ServiceExt};

    use super::*;
    use crate::config::guardrails::{EngineConfig, GuardrailsConfig};
    use crate::guardrails::pipeline::GuardrailPipeline;

    // ── Stub inner service ────────────────────────────────────────────────────

    #[derive(Clone)]
    struct Ok200Svc;

    impl Service<Request<Body>> for Ok200Svc {
        type Response = Response<Body>;
        type Error = Infallible;
        type Future = std::future::Ready<Result<Self::Response, Self::Error>>;

        fn poll_ready(
            &mut self,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: Request<Body>) -> Self::Future {
            std::future::ready(Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"choices":[{"message":{"content":"hello"}}]}"#,
                ))
                .unwrap()))
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn secret_detection_config() -> GuardrailsConfig {
        GuardrailsConfig {
            enabled: true,
            fail_mode: "open".into(),
            timeout: "500ms".into(),
            streaming_mode: "async_audit".into(),
            engines: vec![EngineConfig {
                engine_type: "builtin_secret_detection".into(),
                phase: "pre_request".into(),
                rules: vec![],
                action: Some("block".into()),
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

    // ── Tests: from_config ────────────────────────────────────────────────────

    #[test]
    fn test_guardrail_layer_from_config_disabled() {
        let config = GuardrailsConfig {
            enabled: false,
            ..GuardrailsConfig::default()
        };
        let result = GuardrailLayer::from_config(&config);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "disabled config should return None"
        );
    }

    #[test]
    fn test_guardrail_layer_from_config_enabled() {
        let config = secret_detection_config();
        let result = GuardrailLayer::from_config(&config);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_some(),
            "enabled config should return Some"
        );
    }

    // ── Tests: is_streaming_request ───────────────────────────────────────────

    #[test]
    fn test_is_streaming_request_true() {
        let body =
            br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
        assert!(is_streaming_request(body));
    }

    #[test]
    fn test_is_streaming_request_false() {
        let body =
            br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":false}"#;
        assert!(!is_streaming_request(body));
    }

    #[test]
    fn test_is_streaming_request_no_field() {
        let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;
        assert!(!is_streaming_request(body));
    }

    // ── Tests: guardrail blocks request with secret ───────────────────────────

    #[tokio::test]
    async fn test_guardrail_blocks_request_with_secret() {
        let config = secret_detection_config();
        let layer = GuardrailLayer::from_config(&config)
            .expect("config valid")
            .expect("guardrails enabled");

        let mut svc = layer.layer(Ok200Svc);

        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "My key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456789"}]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        svc.ready().await.unwrap();
        let resp = svc.call(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::FORBIDDEN,
            "request with API secret should be blocked with 403"
        );

        let resp_bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&resp_bytes).unwrap();
        assert_eq!(json["error"], "forbidden");
    }

    // ── Tests: clean request passes through ───────────────────────────────────

    #[tokio::test]
    async fn test_guardrail_passes_clean_request() {
        let config = secret_detection_config();
        let layer = GuardrailLayer::from_config(&config)
            .expect("config valid")
            .expect("guardrails enabled");

        let mut svc = layer.layer(Ok200Svc);

        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What is the capital of France?"}]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        svc.ready().await.unwrap();
        let resp = svc.call(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "clean request should pass with 200"
        );
    }

    // ── Tests: extract_content_from_body ─────────────────────────────────────

    #[test]
    fn test_extract_content_from_messages() {
        let body = br#"{"messages":[{"role":"user","content":"Hello there"}]}"#;
        let content = extract_content_from_body(body);
        assert_eq!(content, "Hello there");
    }

    #[test]
    fn test_extract_content_fallback_raw() {
        let body = b"not valid json";
        let content = extract_content_from_body(body);
        assert_eq!(content, "not valid json");
    }

    // ── Tests: pipeline construction ─────────────────────────────────────────

    #[test]
    fn test_guardrail_pipeline_from_enabled_config() {
        let config = secret_detection_config();
        let pipeline = GuardrailPipeline::from_config(&config);
        assert!(pipeline.is_ok());
    }

    // ── Tests: GuardrailLayer pipeline timeout field ──────────────────────────

    #[test]
    fn test_guardrail_layer_new_with_pipeline() {
        let config = GuardrailsConfig {
            enabled: true,
            fail_mode: "open".into(),
            timeout: "100ms".into(),
            streaming_mode: "async_audit".into(),
            engines: vec![],
        };
        let pipeline = Arc::new(GuardrailPipeline::from_config(&config).unwrap());
        let layer = GuardrailLayer::new(pipeline);
        // Verify debug output includes struct name.
        let dbg = format!("{layer:?}");
        assert!(dbg.contains("GuardrailLayer"));
    }
}
