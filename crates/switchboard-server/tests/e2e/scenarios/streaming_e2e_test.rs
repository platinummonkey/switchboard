//! E2E scenario: streaming_e2e_test
//!
//! End-to-end tests for Server-Sent Events (SSE) streaming over a real TCP
//! connection.
//!
//! These tests verify that:
//!  - The proxy sets the correct `Content-Type: text/event-stream` header.
//!  - Individual SSE data chunks arrive and are collected in order.
//!  - The `[DONE]` sentinel is stripped before being returned to the caller.
//!  - Anthropic streaming events are forwarded correctly.

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::assertions;
    use crate::client::TestClient;
    use crate::harness::TestHarnessBuilder;
    use crate::mocks::{anthropic, openai};

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn openai_stream_body() -> serde_json::Value {
        json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "stream test"}]
        })
    }

    // ── Tests ──────────────────────────────────────────────────────────────────

    /// A streaming OpenAI response must carry `Content-Type: text/event-stream`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_openai_sse_content_type_header() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;
        let openai_mock = harness.mocks.openai.as_ref().unwrap();

        let chunk = r#"{"choices":[{"delta":{"content":"hi"},"index":0,"finish_reason":null}]}"#;
        openai::mock_chat_streaming(&[chunk])
            .mount(openai_mock)
            .await;

        let resp = harness
            .client
            .chat_completions_stream(openai_stream_body())
            .await;

        assert_eq!(resp.status().as_u16(), 200);
        assertions::assert_sse_content_type(&resp);
    }

    /// All SSE data events from a 3-chunk stream must be collected.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_openai_sse_data_arrives() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;
        let openai_mock = harness.mocks.openai.as_ref().unwrap();

        let chunk_a = r#"{"choices":[{"delta":{"content":"a"},"index":0,"finish_reason":null}]}"#;
        let chunk_b = r#"{"choices":[{"delta":{"content":"b"},"index":0,"finish_reason":null}]}"#;
        let chunk_c = r#"{"choices":[{"delta":{"content":"c"},"index":0,"finish_reason":"stop"}]}"#;

        openai::mock_chat_streaming(&[chunk_a, chunk_b, chunk_c])
            .mount(openai_mock)
            .await;

        let resp = harness
            .client
            .chat_completions_stream(openai_stream_body())
            .await;

        assert_eq!(resp.status().as_u16(), 200);

        let events = TestClient::collect_sse(resp).await;

        assert_eq!(
            events.len(),
            3,
            "expected exactly 3 SSE data events (one per chunk), got: {:?}",
            events
        );
        assert!(
            events.contains(&chunk_a.to_string()),
            "event 'a' must be present"
        );
        assert!(
            events.contains(&chunk_b.to_string()),
            "event 'b' must be present"
        );
        assert!(
            events.contains(&chunk_c.to_string()),
            "event 'c' must be present"
        );
    }

    /// SSE chunks must be delivered and collected in the same order they were
    /// emitted by the upstream.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_openai_sse_chunks_in_order() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;
        let openai_mock = harness.mocks.openai.as_ref().unwrap();

        let chunks = &[
            r#"{"choices":[{"delta":{"content":"first"},"index":0,"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"content":"second"},"index":0,"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{"content":"third"},"index":0,"finish_reason":"stop"}]}"#,
        ];

        openai::mock_chat_streaming(chunks).mount(openai_mock).await;

        let resp = harness
            .client
            .chat_completions_stream(openai_stream_body())
            .await;

        assert_eq!(resp.status().as_u16(), 200);

        let events = TestClient::collect_sse(resp).await;

        assert_eq!(
            events.len(),
            3,
            "expected exactly 3 ordered SSE events, got: {:?}",
            events
        );

        // Parse each event and verify the content field preserves order.
        let contents: Vec<&str> = events
            .iter()
            .map(|e| {
                let v: serde_json::Value =
                    serde_json::from_str(e).expect("each SSE event must be valid JSON");
                // Extract content as a &str from the owned Value — we compare
                // the raw event strings directly below so just validate shape.
                let _ = v["choices"][0]["delta"]["content"]
                    .as_str()
                    .expect("delta.content must be a string");
                e.as_str()
            })
            .collect();

        assert_eq!(contents[0], chunks[0], "first chunk must arrive first");
        assert_eq!(contents[1], chunks[1], "second chunk must arrive second");
        assert_eq!(contents[2], chunks[2], "third chunk must arrive third");
    }

    /// Anthropic streaming responses must produce SSE events containing
    /// `content_block_delta` payloads with the text delta.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_anthropic_sse_events() {
        let harness = TestHarnessBuilder::new().with_anthropic().build().await;
        let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

        anthropic::mock_messages_streaming("streaming text")
            .mount(anthropic_mock)
            .await;

        let resp = harness
            .client
            .anthropic_messages_stream(json!({
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "stream?"}]
            }))
            .await;

        assert_eq!(resp.status().as_u16(), 200);

        let events = TestClient::collect_sse(resp).await;

        assert!(
            !events.is_empty(),
            "expected at least one SSE data event from the Anthropic streaming mock"
        );

        // At least one event must be a content_block_delta with our text.
        let has_delta = events.iter().any(|e| {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(e) {
                v["type"] == "content_block_delta" && v["delta"]["text"] == "streaming text"
            } else {
                false
            }
        });
        assert!(
            has_delta,
            "expected a content_block_delta event with text 'streaming text', events: {:?}",
            events
        );
    }

    /// The `[DONE]` SSE sentinel must NOT appear in the events collected by
    /// `TestClient::collect_sse`.
    ///
    /// OpenAI (and Ollama) append `data: [DONE]` to signal end-of-stream.
    /// The helper must strip this so tests never see it as a data payload.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_done_sentinel_not_included() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;
        let openai_mock = harness.mocks.openai.as_ref().unwrap();

        let chunk = r#"{"choices":[{"delta":{"content":"hi"},"index":0,"finish_reason":"stop"}]}"#;

        // `mock_chat_streaming` appends `data: [DONE]\n\n` automatically.
        openai::mock_chat_streaming(&[chunk])
            .mount(openai_mock)
            .await;

        let resp = harness
            .client
            .chat_completions_stream(openai_stream_body())
            .await;

        assert_eq!(resp.status().as_u16(), 200);

        let events = TestClient::collect_sse(resp).await;

        // `[DONE]` must have been stripped.
        assert!(
            !events.iter().any(|e| e == "[DONE]"),
            "[DONE] sentinel must be filtered from collected SSE events, got: {:?}",
            events
        );

        // The real chunk must still be present.
        assert!(
            events.contains(&chunk.to_string()),
            "the real data chunk must survive [DONE] filtering, got: {:?}",
            events
        );
    }
}
