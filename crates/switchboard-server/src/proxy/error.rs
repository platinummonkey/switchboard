//! Proxy layer errors and their HTTP response mappings.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

/// Errors that can occur in the proxy request pipeline.
#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("no provider available for model: {0}")]
    NoProvider(String),

    #[error("no key available for provider: {0}")]
    NoKey(String),

    #[error("upstream error: {0}")]
    Upstream(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ProxyError::InvalidRequest(msg) => {
                tracing::warn!(error = %self, "proxy: invalid request");
                (
                    StatusCode::BAD_REQUEST,
                    format!(
                        "{{\"error\":{{\"message\":\"{msg}\",\"type\":\"invalid_request_error\"}}}}"
                    ),
                )
            }
            ProxyError::NoProvider(model) => {
                tracing::warn!(model, "proxy: no provider for model");
                (
                    StatusCode::BAD_REQUEST,
                    format!(
                        "{{\"error\":{{\"message\":\"No provider available for model: {model}\",\"type\":\"invalid_request_error\"}}}}"
                    ),
                )
            }
            ProxyError::NoKey(provider) => {
                tracing::error!(provider, "proxy: no key available");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "{{\"error\":{{\"message\":\"No key available for provider: {provider}\",\"type\":\"service_unavailable\"}}}}"
                    ),
                )
            }
            ProxyError::Upstream(msg) => {
                tracing::error!(error = %self, "proxy: upstream error");
                (
                    StatusCode::BAD_GATEWAY,
                    format!("{{\"error\":{{\"message\":\"{msg}\",\"type\":\"upstream_error\"}}}}"),
                )
            }
            ProxyError::Serialization(e) => {
                tracing::error!(error = %e, "proxy: serialization error");
                (
                    StatusCode::BAD_REQUEST,
                    format!(
                        "{{\"error\":{{\"message\":\"Serialization error: {e}\",\"type\":\"invalid_request_error\"}}}}"
                    ),
                )
            }
        };

        (status, [("content-type", "application/json")], message).into_response()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    use super::*;

    #[test]
    fn test_proxy_error_display_invalid_request() {
        let e = ProxyError::InvalidRequest("missing model field".into());
        assert!(e.to_string().contains("missing model field"));
    }

    #[test]
    fn test_proxy_error_display_no_provider() {
        let e = ProxyError::NoProvider("gpt-99".into());
        assert!(e.to_string().contains("gpt-99"));
    }

    #[test]
    fn test_proxy_error_display_no_key() {
        let e = ProxyError::NoKey("anthropic".into());
        assert!(e.to_string().contains("anthropic"));
    }

    #[test]
    fn test_proxy_error_display_upstream() {
        let e = ProxyError::Upstream("connection refused".into());
        assert!(e.to_string().contains("connection refused"));
    }

    #[test]
    fn test_invalid_request_returns_400() {
        let resp = ProxyError::InvalidRequest("bad".into()).into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_no_provider_returns_400() {
        let resp = ProxyError::NoProvider("unknown-model".into()).into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_no_key_returns_503() {
        let resp = ProxyError::NoKey("anthropic".into()).into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_upstream_error_returns_502() {
        let resp = ProxyError::Upstream("timeout".into()).into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn test_serialization_error_returns_400() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let resp = ProxyError::Serialization(json_err).into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
