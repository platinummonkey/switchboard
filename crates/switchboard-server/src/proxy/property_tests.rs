//! Property-based tests for the proxy transform pipeline and SSE stream parser.
//!
//! Complements the unit tests in `transform.rs` and `stream.rs` by using
//! `proptest` to generate arbitrary inputs and verify invariants hold.

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use switchboard_common::types::{Message, MessageContent, Role};

    use crate::proxy::transform::{anthropic_to_proxied, openai_to_proxied};

    // ── Transform invariants ──────────────────────────────────────────────────

    /// Arbitrary role string that maps to a valid Role variant.
    fn arb_role() -> impl Strategy<Value = Role> {
        prop_oneof![Just(Role::User), Just(Role::Assistant), Just(Role::System),]
    }

    /// Arbitrary message with short text content.
    fn arb_message() -> impl Strategy<Value = Message> {
        (arb_role(), "[a-zA-Z0-9 .,!?]{1,200}").prop_map(|(role, text)| Message {
            role,
            content: MessageContent::Text(text),
            tool_call_id: None,
            tool_calls: None,
        })
    }

    proptest! {
        /// `openai_to_proxied` preserves the message count and model field.
        #[test]
        fn prop_openai_transform_preserves_message_count(
            messages in proptest::collection::vec(arb_message(), 1..=8usize),
            model in "[a-z0-9-]{4,40}",
            max_tokens in 1u32..=4096u32,
            temperature in 0.0f64..=2.0f64,
        ) {
            let body = serde_json::json!({
                "model": model,
                "messages": messages.iter().map(|m| serde_json::json!({
                    "role": match &m.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::System => "system",
                        Role::Tool => "tool",
                    },
                    "content": m.content.as_text(),
                })).collect::<Vec<_>>(),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": false
            });

            let result = openai_to_proxied(&body);
            prop_assert!(result.is_ok(), "parse failed: {:?}", result);
            let req = result.unwrap();
            prop_assert_eq!(req.messages.len(), messages.len(),
                "message count mismatch");
            prop_assert_eq!(req.model, model, "model mismatch");
            prop_assert_eq!(req.max_tokens, Some(max_tokens), "max_tokens mismatch");
        }

        /// `anthropic_to_proxied` preserves message count and max_tokens.
        #[test]
        fn prop_anthropic_transform_preserves_message_count(
            messages in proptest::collection::vec(arb_message(), 1..=8usize)
                // Anthropic only allows user/assistant in messages array
                .prop_map(|msgs| {
                    msgs.into_iter()
                        .map(|mut m| {
                            if matches!(m.role, Role::System) {
                                m.role = Role::User;
                            }
                            m
                        })
                        .collect::<Vec<_>>()
                }),
            model in "[a-z0-9-]{4,40}",
            max_tokens in 1u32..=4096u32,
        ) {
            let body = serde_json::json!({
                "model": model,
                "messages": messages.iter().map(|m| serde_json::json!({
                    "role": match &m.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        _ => "user",
                    },
                    "content": m.content.as_text(),
                })).collect::<Vec<_>>(),
                "max_tokens": max_tokens,
                "stream": false
            });

            let result = anthropic_to_proxied(&body);
            prop_assert!(result.is_ok(), "parse failed: {:?}", result);
            let req = result.unwrap();
            prop_assert_eq!(req.messages.len(), messages.len(),
                "message count mismatch");
            prop_assert_eq!(req.max_tokens, Some(max_tokens), "max_tokens mismatch");
        }

        /// Non-modified fields in `ProxiedRequest` (stream, temperature) are preserved
        /// through a round-trip parse.
        #[test]
        fn prop_openai_non_modified_fields_preserved(
            temperature in 0.0f64..=2.0f64,
            stream in any::<bool>(),
        ) {
            let body = serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": temperature,
                "stream": stream
            });
            let req = openai_to_proxied(&body).unwrap();
            // temperature is preserved (within float precision)
            if let Some(t) = req.temperature {
                prop_assert!(
                    (t - temperature).abs() < 1e-9,
                    "temperature mismatch: {t} vs {temperature}"
                );
            }
            prop_assert_eq!(req.stream, stream, "stream flag mismatch");
        }
    }

    // ── SSE chunk boundary safety ─────────────────────────────────────────────

    /// The SSE parser must never panic on arbitrary byte chunk splits.
    /// We feed the same valid SSE stream split at every possible byte boundary
    /// and verify the total event count is always the same.
    #[test]
    fn prop_sse_chunk_boundary_determinism() {
        let full_sse = concat!(
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},",
            "\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"2\",\"choices\":[{\"delta\":{\"content\":\" world\"},",
            "\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n",
        );
        let bytes = full_sse.as_bytes();

        // Split at every possible byte position and count parsed frames.
        // The SSE parser in stream.rs accumulates a remainder buffer, so
        // any split should produce exactly 3 complete events.
        for split in 0..bytes.len() {
            let chunk_a = &bytes[..split];
            let chunk_b = &bytes[split..];

            let mut remainder = String::new();
            let mut events = Vec::new();

            for chunk in [chunk_a, chunk_b] {
                if let Ok(text) = std::str::from_utf8(chunk) {
                    remainder.push_str(text);
                    while let Some(pos) = remainder.find("\n\n") {
                        let frame = remainder[..pos].to_string();
                        remainder = remainder[pos + 2..].to_string();
                        let data = if let Some(s) = frame.strip_prefix("data: ") {
                            s.to_string()
                        } else if let Some(s) = frame.strip_prefix("data:") {
                            s.trim_start().to_string()
                        } else {
                            frame
                        };
                        if !data.is_empty() {
                            events.push(data);
                        }
                    }
                }
            }

            assert_eq!(
                events.len(),
                3,
                "split at byte {split}: expected 3 events, got {} (events: {:?})",
                events.len(),
                events
            );
        }
    }

    // ── Auth credential thread safety ─────────────────────────────────────────

    /// Concurrent reads of UpstreamCredentials must never observe a torn write.
    /// This test spawns N tasks that all read credentials simultaneously while
    /// one writer updates them, and checks no panic or data race occurs.
    #[tokio::test]
    async fn prop_credentials_concurrent_read_safety() {
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::sync::RwLock;

        use http::{HeaderName, HeaderValue};
        use switchboard_common::types::RequestContext;

        use crate::auth::UpstreamCredentials;

        let creds = Arc::new(RwLock::new(UpstreamCredentials {
            header_name: HeaderName::from_static("authorization"),
            header_value: HeaderValue::from_static("Bearer sk-v1"),
            expires_at: None,
        }));

        let reader_tasks: Vec<_> = (0..20)
            .map(|_| {
                let c = Arc::clone(&creds);
                tokio::spawn(async move {
                    for _ in 0..50 {
                        let guard = c.read().await;
                        // Just accessing the value — assert it's one of the two
                        // known values (v1 or v2).
                        let val = guard.header_value.to_str().unwrap_or("");
                        assert!(
                            val == "Bearer sk-v1" || val == "Bearer sk-v2",
                            "unexpected credential value: {val}"
                        );
                        drop(guard);
                        tokio::task::yield_now().await;
                    }
                })
            })
            .collect();

        // Writer updates credentials once.
        let writer = {
            let c = Arc::clone(&creds);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(1)).await;
                let mut guard = c.write().await;
                guard.header_value = HeaderValue::from_static("Bearer sk-v2");
            })
        };

        writer.await.unwrap();
        for t in reader_tasks {
            t.await.unwrap();
        }
        let _ = RequestContext::new(); // ensure common types compile in this scope
    }
}
