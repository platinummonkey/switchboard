//! E2E scenario: auth_e2e_test
//!
//! End-to-end tests for client→server authentication.
//!
//! These tests exercise auth through the full running TCP server stack — the
//! server starts with a static-key validator configured for `"test-api-key"`.
//! Requests without a valid bearer token must be rejected with 401 before
//! reaching any provider.

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::harness::TestHarnessBuilder;
    use crate::mocks::openai;

    // ── Shared request body ────────────────────────────────────────────────────

    fn openai_body() -> serde_json::Value {
        json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        })
    }

    // ── Tests ──────────────────────────────────────────────────────────────────

    /// A request with no Authorization header must be rejected with 401.
    ///
    /// The auth middleware must fire before any upstream proxy logic so that
    /// unauthenticated callers never reach a provider.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_missing_auth_header_returns_401() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;

        let resp = harness.client.chat_no_auth(openai_body()).await;

        assert_eq!(
            resp.status().as_u16(),
            401,
            "missing Authorization header must return 401"
        );
    }

    /// A request with an unrecognised API key must be rejected with 401.
    ///
    /// The static-key validator only accepts `"test-api-key"`.  Any other
    /// token — even a syntactically valid bearer string — must be refused.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_invalid_api_key_returns_401() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;

        let resp = harness
            .client
            .chat_with_key(openai_body(), "wrong-key")
            .await;

        assert_eq!(
            resp.status().as_u16(),
            401,
            "invalid API key must return 401"
        );
    }

    /// A request with the valid API key (`"test-api-key"`) must be proxied
    /// through to the upstream and return 200.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_valid_api_key_returns_ok() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;
        let openai_mock = harness.mocks.openai.as_ref().unwrap();

        openai::mock_chat_ok("gpt-4o", "authenticated!")
            .mount(openai_mock)
            .await;

        // `chat_completions` sends the built-in "test-api-key" bearer token.
        let resp = harness.client.chat_completions(openai_body()).await;

        assert_eq!(
            resp.status().as_u16(),
            200,
            "valid API key must return 200 from the upstream mock"
        );

        let body: serde_json::Value = resp.json().await.expect("response must be JSON");
        assert_eq!(
            body["choices"][0]["message"]["content"], "authenticated!",
            "response content must match the mock reply"
        );
    }

    /// Auth is enforced on ALL routes, not just the OpenAI endpoint.
    ///
    /// Sends a request to `/v1/chat/completions` with a Claude model name but
    /// without any Authorization header.  Even though the anthropic provider
    /// is configured, the auth middleware must reject the request with 401
    /// before the request is forwarded.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_e2e_auth_applies_to_anthropic_endpoint() {
        // Build a harness with the Anthropic provider enabled so the server
        // would otherwise be capable of routing this request.
        let harness = TestHarnessBuilder::new().with_anthropic().build().await;

        // Call `chat_no_auth` — no Authorization header is sent.  We use a
        // Claude model name to confirm the auth check fires regardless of
        // which model/provider would handle the request.
        let resp = harness
            .client
            .chat_no_auth(json!({
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .await;

        assert_eq!(
            resp.status().as_u16(),
            401,
            "auth must be enforced on all endpoints; unauthenticated request with claude model must return 401"
        );
    }
}
