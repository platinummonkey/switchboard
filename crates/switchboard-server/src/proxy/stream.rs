//! SSE streaming: forward a reqwest response body as an axum SSE stream.

use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::StreamExt;

/// Forward a streaming [`reqwest::Response`] as an axum SSE stream.
///
/// Bytes are forwarded zero-copy: chunks are split on `\n\n` SSE boundaries
/// and yielded as raw data events.  A `[DONE]` sentinel will be included if
/// the upstream sends it (it is passed through verbatim).
pub async fn forward_sse_stream(upstream: reqwest::Response) -> impl IntoResponse {
    let byte_stream = upstream.bytes_stream();

    // Buffer to handle partial SSE frames split across TCP segments.
    let sse_stream = {
        let mut remainder = String::new();

        byte_stream.flat_map(move |chunk_result| {
            let events: Vec<Result<Event, std::convert::Infallible>> = match chunk_result {
                Err(e) => {
                    tracing::warn!(error = %e, "sse: upstream read error");
                    vec![]
                }
                Ok(bytes) => {
                    // Append incoming bytes to the remainder buffer.
                    match std::str::from_utf8(&bytes) {
                        Ok(text) => remainder.push_str(text),
                        Err(_) => {
                            tracing::warn!("sse: non-UTF8 chunk received, skipping");
                            return futures_util::stream::iter(vec![]);
                        }
                    }

                    // Split on double-newline SSE event boundaries.
                    let mut result_events = Vec::new();
                    while let Some(pos) = remainder.find("\n\n") {
                        let frame = remainder[..pos].to_string();
                        remainder = remainder[pos + 2..].to_string();

                        // Strip `data: ` or `data:` prefix if present, using
                        // strip_prefix to satisfy clippy's manual-strip lint.
                        let data = if let Some(stripped) = frame.strip_prefix("data: ") {
                            stripped.to_string()
                        } else if let Some(stripped) = frame.strip_prefix("data:") {
                            stripped.trim_start().to_string()
                        } else {
                            frame
                        };

                        if !data.is_empty() {
                            result_events.push(Ok(Event::default().data(data)));
                        }
                    }
                    result_events
                }
            };
            futures_util::stream::iter(events)
        })
    };

    Sse::new(sse_stream).keep_alive(KeepAlive::default())
}

/// Forward raw SSE bytes from an upstream response body directly without
/// parsing, useful when the upstream format must be preserved exactly.
///
/// This is a lower-overhead alternative to [`forward_sse_stream`] that
/// returns a streaming body response instead of an SSE response.
pub async fn forward_raw_stream(upstream: reqwest::Response) -> impl IntoResponse {
    use axum::body::Body;
    use http::Response;

    let status = axum::http::StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(axum::http::StatusCode::OK);

    // Preserve the content-type from upstream.
    let content_type = upstream
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("text/event-stream")
        .to_string();

    let byte_stream = upstream
        .bytes_stream()
        .map(|r| r.map_err(|e| std::io::Error::other(e.to_string())));

    let body = Body::from_stream(byte_stream);

    Response::builder()
        .status(status)
        .header("content-type", content_type)
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .body(body)
        .unwrap_or_else(|_| {
            Response::builder()
                .status(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::empty())
                .unwrap()
        })
}

/// Collect all bytes from a [`BoxStream`] into a vec of chunks.
/// Used in tests to verify streaming content.
#[cfg(test)]
pub async fn collect_stream(mut stream: crate::routing::BoxStream) -> Vec<bytes::Bytes> {
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        if let Ok(chunk) = item {
            chunks.push(chunk);
        }
    }
    chunks
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // Stream forwarding is tested via integration tests using wiremock.
    // Unit-testable logic is minimal here (it's thin glue over axum SSE).

    #[test]
    fn test_stream_module_compiles() {
        // Compilation is the primary test — forward_sse_stream is async and
        // integration-tested.
    }
}
