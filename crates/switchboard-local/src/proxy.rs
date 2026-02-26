//! Request forwarding logic for `switchboard-local`.
//!
//! [`forward_request`] is the core function called by every proxied route.  It:
//! 1. Obtains the current auth header from [`LocalAuthManager`].
//! 2. Injects `X-Switchboard-User` / `X-Switchboard-Team` identity headers.
//! 3. Applies model overrides to the JSON request body.
//! 4. Optionally injects `X-Switchboard-Model` with the default model.
//! 5. Forwards the request to the upstream `switchboard-server`.
//! 6. Streams (`text/event-stream`) or buffers the response back to the caller.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::StreamExt;
use tracing::instrument;

use crate::auth::LocalAuthManager;
use crate::config::LocalConfig;
use crate::model_prefs::ModelPrefs;

/// Hop-by-hop headers that must NOT be forwarded to the upstream server.
static HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "transfer-encoding",
    "te",
    "trailer",
    "upgrade",
    "proxy-connection",
    "proxy-authenticate",
    "proxy-authorization",
];

/// Shared state threaded through every axum handler.
pub struct LocalServerState {
    pub config: Arc<LocalConfig>,
    pub auth_manager: Arc<LocalAuthManager>,
    pub model_prefs: Arc<ModelPrefs>,
    pub client: reqwest::Client,
}

/// Forward an incoming request to `switchboard-server` and relay the response.
///
/// This function is the single code-path for all proxied routes.
#[instrument(skip_all, fields(method = %method, path = %path))]
pub async fn forward_request(
    state: Arc<LocalServerState>,
    method: Method,
    path: &str,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // 1. Obtain auth header — 401 if unavailable.
    let (auth_name, auth_value) = match state.auth_manager.get_header().await {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!(error = %e, "local proxy: auth header unavailable");
            return (
                StatusCode::UNAUTHORIZED,
                format!("authentication unavailable: {e}"),
            )
                .into_response();
        }
    };

    // 2. Possibly rewrite the body (model overrides + default model injection).
    let (final_body, is_streaming) = rewrite_body(&state.model_prefs, &body);

    // 3. Build the upstream URL.
    let url = format!("{}{}", state.config.server.url.trim_end_matches('/'), path);

    // 4. Build the outgoing request.
    let mut req_builder = state.client.request(method.clone(), &url);

    // Forward headers, stripping hop-by-hop ones plus content-length (reqwest
    // will recompute it from the rewritten body).
    for (name, value) in &headers {
        let lower = name.as_str().to_lowercase();
        if HOP_BY_HOP.contains(&lower.as_str()) {
            continue;
        }
        // Skip Host; reqwest will set it from the URL.
        if lower == "host" {
            continue;
        }
        // Skip content-length; reqwest will set the correct value after body
        // rewriting (the length may have changed due to model override).
        if lower == "content-length" {
            continue;
        }
        req_builder = req_builder.header(name.as_str(), value.as_bytes());
    }

    // Auth header (overrides whatever the client sent).
    req_builder = req_builder.header(auth_name.as_str(), auth_value.as_str());

    // Identity headers.
    if let Some(user) = &state.config.identity.user {
        req_builder = req_builder.header("X-Switchboard-User", user.as_str());
    } else if let Ok(os_user) = std::env::var("USER") {
        if !os_user.is_empty() {
            req_builder = req_builder.header("X-Switchboard-User", os_user.as_str());
        }
    }
    if let Some(team) = &state.config.identity.team {
        req_builder = req_builder.header("X-Switchboard-Team", team.as_str());
    }

    // Inject default model header if not already present.
    if !headers.contains_key("x-switchboard-model") {
        req_builder = req_builder.header("X-Switchboard-Model", state.model_prefs.default_model());
    }

    req_builder = req_builder.body(final_body);

    // 5. Send to upstream.
    let upstream_resp = match req_builder.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, url = %url, "local proxy: upstream request failed");
            return (
                StatusCode::BAD_GATEWAY,
                format!("upstream request failed: {e}"),
            )
                .into_response();
        }
    };

    let status =
        StatusCode::from_u16(upstream_resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    // Collect upstream response headers (strip hop-by-hop).
    let mut resp_headers = HeaderMap::new();
    for (name, value) in upstream_resp.headers() {
        let lower = name.as_str().to_lowercase();
        if HOP_BY_HOP.contains(&lower.as_str()) {
            continue;
        }
        if let (Ok(hn), Ok(hv)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            resp_headers.insert(hn, hv);
        }
    }

    // Check if the response is SSE / streaming.
    let resp_is_sse = upstream_resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("text/event-stream"))
        .unwrap_or(false);

    // 6. Stream or buffer the response body.
    if is_streaming || resp_is_sse {
        // Stream back via chunked transfer.
        let stream = upstream_resp.bytes_stream().map(|chunk| {
            chunk.map_err(|e| {
                tracing::error!(error = %e, "local proxy: error reading upstream stream chunk");
                std::io::Error::other(e)
            })
        });
        let body = Body::from_stream(stream);
        (status, resp_headers, body).into_response()
    } else {
        // Buffer the whole body.
        let resp_bytes = match upstream_resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                tracing::error!(error = %e, "local proxy: failed to read upstream response body");
                return (
                    StatusCode::BAD_GATEWAY,
                    format!("failed to read upstream response: {e}"),
                )
                    .into_response();
            }
        };
        (status, resp_headers, resp_bytes).into_response()
    }
}

/// Attempt to parse `body` as JSON, apply model overrides, and return the
/// (possibly rewritten) bytes together with a flag indicating whether the
/// client requested a streaming response.
///
/// If the body is not valid JSON, it is returned as-is and `is_streaming` is
/// `false`.
fn rewrite_body(model_prefs: &ModelPrefs, body: &Bytes) -> (Bytes, bool) {
    if body.is_empty() {
        return (body.clone(), false);
    }

    match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(mut json) => {
            // Apply model override (if configured).
            model_prefs.apply_to_body(&mut json);

            // Detect streaming request.
            let is_streaming = json
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // Re-serialise only if necessary (avoid re-encoding if unchanged).
            match serde_json::to_vec(&json) {
                Ok(bytes) => (Bytes::from(bytes), is_streaming),
                Err(_) => (body.clone(), is_streaming),
            }
        }
        Err(_) => {
            // Not JSON — pass through unchanged.
            (body.clone(), false)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use std::collections::HashMap;

    fn make_prefs_with_override() -> ModelPrefs {
        let mut overrides = HashMap::new();
        overrides.insert("gpt-4".into(), "claude-sonnet-4-20250514".into());
        ModelPrefs::from_config(&ModelConfig {
            default: "claude-sonnet-4-20250514".into(),
            overrides,
        })
    }

    #[test]
    fn test_rewrite_body_applies_override() {
        let prefs = make_prefs_with_override();
        let body = Bytes::from(r#"{"model":"gpt-4","messages":[]}"#);
        let (out, _) = rewrite_body(&prefs, &body);
        let val: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(val["model"].as_str(), Some("claude-sonnet-4-20250514"));
    }

    #[test]
    fn test_rewrite_body_detects_streaming() {
        let prefs = make_prefs_with_override();
        let body = Bytes::from(r#"{"model":"gpt-4","stream":true}"#);
        let (_, streaming) = rewrite_body(&prefs, &body);
        assert!(streaming);
    }

    #[test]
    fn test_rewrite_body_non_json_passthrough() {
        let prefs = make_prefs_with_override();
        let raw = Bytes::from("not json at all");
        let (out, streaming) = rewrite_body(&prefs, &raw);
        assert_eq!(out, raw);
        assert!(!streaming);
    }

    #[test]
    fn test_rewrite_body_empty_passthrough() {
        let prefs = make_prefs_with_override();
        let (out, streaming) = rewrite_body(&prefs, &Bytes::new());
        assert!(out.is_empty());
        assert!(!streaming);
    }
}
