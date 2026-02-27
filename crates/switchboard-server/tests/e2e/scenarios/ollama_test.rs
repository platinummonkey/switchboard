//! E2E scenario: ollama_test
//!
//! End-to-end tests for the Ollama provider integration.
//!
//! Ollama runs in OpenAI-compatible mode (`api_format = "ollama"`) so the
//! switchboard server translates between the OpenAI wire format and Ollama's
//! `/v1/chat/completions` endpoint transparently.

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::harness::TestHarnessBuilder;
    use crate::mocks::ollama;

    /// Chat completion with llama3.2 returns 200 and the expected content.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_ollama_llama3_chat() {
        let harness = TestHarnessBuilder::new().with_ollama().build().await;
        let ollama_mock = harness.mocks.ollama.as_ref().unwrap();

        ollama::mock_chat_ok("llama3.2", "42")
            .mount(ollama_mock)
            .await;

        let resp = harness
            .client
            .chat_completions(json!({
                "model": "llama3.2",
                "messages": [{"role": "user", "content": "Answer?"}]
            }))
            .await;

        assert_eq!(resp.status().as_u16(), 200);

        let body: serde_json::Value = resp.json().await.expect("response body must be JSON");
        assert_eq!(
            body["choices"][0]["message"]["content"], "42",
            "expected content '42' from llama3.2 mock"
        );
    }

    /// Chat completion with mistral returns 200 and the expected content.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_ollama_mistral_chat() {
        let harness = TestHarnessBuilder::new().with_ollama().build().await;
        let ollama_mock = harness.mocks.ollama.as_ref().unwrap();

        ollama::mock_chat_ok("mistral", "42")
            .mount(ollama_mock)
            .await;

        let resp = harness
            .client
            .chat_completions(json!({
                "model": "mistral",
                "messages": [{"role": "user", "content": "Answer?"}]
            }))
            .await;

        assert_eq!(resp.status().as_u16(), 200);

        let body: serde_json::Value = resp.json().await.expect("response body must be JSON");
        assert_eq!(
            body["choices"][0]["message"]["content"], "42",
            "expected content '42' from mistral mock"
        );
    }

    /// Streaming request receives SSE events from the Ollama provider.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_ollama_streaming() {
        let harness = TestHarnessBuilder::new().with_ollama().build().await;
        let ollama_mock = harness.mocks.ollama.as_ref().unwrap();

        let chunk_a =
            r#"{"choices":[{"delta":{"content":"hello"},"index":0,"finish_reason":null}]}"#;
        let chunk_b =
            r#"{"choices":[{"delta":{"content":" world"},"index":0,"finish_reason":"stop"}]}"#;

        ollama::mock_chat_streaming(&[chunk_a, chunk_b])
            .mount(ollama_mock)
            .await;

        let resp = harness
            .client
            .chat_completions_stream(json!({
                "model": "llama3.2",
                "messages": [{"role": "user", "content": "stream?"}]
            }))
            .await;

        assert_eq!(resp.status().as_u16(), 200);

        let events = crate::client::TestClient::collect_sse(resp).await;
        assert!(
            !events.is_empty(),
            "expected at least one SSE data event from the Ollama streaming mock"
        );
    }

    /// The response from the Ollama provider is forwarded in OpenAI format
    /// (`choices[0].message.content`), not in a native Ollama format.
    ///
    /// Ollama in openai-compat mode emits `choices[].message.content` already,
    /// so the proxy must preserve that structure end-to-end.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_ollama_uses_openai_format() {
        let harness = TestHarnessBuilder::new().with_ollama().build().await;
        let ollama_mock = harness.mocks.ollama.as_ref().unwrap();

        ollama::mock_chat_ok("llama3.2", "openai-format-check")
            .mount(ollama_mock)
            .await;

        let resp = harness
            .client
            .chat_completions(json!({
                "model": "llama3.2",
                "messages": [{"role": "user", "content": "format?"}]
            }))
            .await;

        assert_eq!(resp.status().as_u16(), 200);

        let body: serde_json::Value = resp.json().await.expect("response body must be JSON");

        // Must have OpenAI-shaped response: choices[0].message.content.
        assert!(
            body.get("choices").is_some(),
            "response must contain 'choices' field (OpenAI format)"
        );
        assert!(
            body["choices"][0].get("message").is_some(),
            "choices[0] must contain 'message' field (OpenAI format)"
        );
        assert_eq!(
            body["choices"][0]["message"]["content"], "openai-format-check",
            "choices[0].message.content must match expected value"
        );
    }
}
