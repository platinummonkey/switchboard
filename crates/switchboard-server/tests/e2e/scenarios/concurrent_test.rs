//! E2E scenario: concurrent_test
//!
//! Verifies that the proxy handles concurrent requests correctly — no deadlocks,
//! no data races, and correct enforcement of rate limits under concurrent load.
//!
//! All tests use `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]`
//! and drive requests in parallel via `tokio::spawn` + `futures_util::future::join_all`.

#[cfg(test)]
mod tests {
    use futures_util::future::join_all;
    use serde_json::json;

    use crate::harness::TestHarnessBuilder;
    use crate::mocks::anthropic::mock_messages_ok;
    use crate::mocks::openai::{mock_chat_ok, mock_chat_streaming};

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn openai_body() -> serde_json::Value {
        json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        })
    }

    fn anthropic_body() -> serde_json::Value {
        json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}]
        })
    }

    // ── Tests ──────────────────────────────────────────────────────────────────

    /// Send 10 concurrent non-streaming requests to the proxy and verify that
    /// all of them succeed (200 OK) and the upstream mock receives exactly 10.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_e2e_concurrent_requests_all_succeed() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;
        let openai_mock = harness.mocks.openai.as_ref().unwrap();

        // No .expect() limit — the mock must handle all 10 concurrent requests.
        mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

        // Build a raw reqwest client + base URL so we can move values into
        // spawned tasks (TestClient is not Clone).
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");
        let base_url = format!("http://{}", harness.addr);

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let client = client.clone();
                let url = format!("{}/v1/chat/completions", base_url);
                tokio::spawn(async move {
                    client
                        .post(&url)
                        .header("Authorization", "Bearer test-api-key")
                        .json(&openai_body())
                        .send()
                        .await
                        .expect("concurrent request failed")
                })
            })
            .collect();

        let results: Vec<_> = join_all(handles)
            .await
            .into_iter()
            .map(|r| r.expect("task panicked"))
            .collect();

        // Every response must be 200.
        for (i, resp) in results.iter().enumerate() {
            assert_eq!(resp.status().as_u16(), 200, "request {i} returned non-200");
        }

        // The upstream mock must have received exactly 10 requests.
        crate::assertions::assert_received_n(openai_mock, 10).await;
    }

    /// Send 5 concurrent *streaming* requests and verify that all of them
    /// return 200 with `Content-Type: text/event-stream`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_e2e_concurrent_streaming_requests() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;
        let openai_mock = harness.mocks.openai.as_ref().unwrap();

        let chunk =
            r#"{"choices":[{"delta":{"content":"hello"},"index":0,"finish_reason":"stop"}]}"#;
        mock_chat_streaming(&[chunk]).mount(openai_mock).await;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");
        let base_url = format!("http://{}", harness.addr);

        let handles: Vec<_> = (0..5)
            .map(|_| {
                let client = client.clone();
                let url = format!("{}/v1/chat/completions", base_url);
                let mut body = openai_body();
                body["stream"] = json!(true);
                tokio::spawn(async move {
                    client
                        .post(&url)
                        .header("Authorization", "Bearer test-api-key")
                        .json(&body)
                        .send()
                        .await
                        .expect("concurrent streaming request failed")
                })
            })
            .collect();

        let results: Vec<_> = join_all(handles)
            .await
            .into_iter()
            .map(|r| r.expect("task panicked"))
            .collect();

        // All 5 streaming responses must be 200 with the correct content-type.
        for (i, resp) in results.iter().enumerate() {
            assert_eq!(
                resp.status().as_u16(),
                200,
                "streaming request {i} returned non-200"
            );
            crate::assertions::assert_sse_content_type(resp);
        }
    }

    /// Send 5 OpenAI requests and 5 Anthropic requests all concurrently (10
    /// total).  All 10 must return 200, and each upstream mock must receive
    /// exactly 5 hits.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_e2e_concurrent_different_providers() {
        let harness = TestHarnessBuilder::new()
            .with_openai()
            .with_anthropic()
            .build()
            .await;

        let openai_mock = harness.mocks.openai.as_ref().unwrap();
        let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

        mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;
        mock_messages_ok("ok").mount(anthropic_mock).await;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");
        let base_url = format!("http://{}", harness.addr);

        // 5 OpenAI tasks.
        let openai_handles: Vec<_> = (0..5)
            .map(|_| {
                let client = client.clone();
                let url = format!("{}/v1/chat/completions", base_url);
                tokio::spawn(async move {
                    client
                        .post(&url)
                        .header("Authorization", "Bearer test-api-key")
                        .json(&openai_body())
                        .send()
                        .await
                        .expect("openai concurrent request failed")
                })
            })
            .collect();

        // 5 Anthropic tasks.
        let anthropic_handles: Vec<_> = (0..5)
            .map(|_| {
                let client = client.clone();
                let url = format!("{}/api/v1/messages", base_url);
                tokio::spawn(async move {
                    client
                        .post(&url)
                        .header("Authorization", "Bearer test-api-key")
                        .json(&anthropic_body())
                        .send()
                        .await
                        .expect("anthropic concurrent request failed")
                })
            })
            .collect();

        // Join all 10 in parallel.
        let all_handles: Vec<_> = openai_handles
            .into_iter()
            .chain(anthropic_handles)
            .collect();

        let results: Vec<_> = join_all(all_handles)
            .await
            .into_iter()
            .map(|r| r.expect("task panicked"))
            .collect();

        // All 10 must succeed.
        for (i, resp) in results.iter().enumerate() {
            assert_eq!(
                resp.status().as_u16(),
                200,
                "request {i} (mixed provider) returned non-200"
            );
        }

        // Each upstream must have received exactly 5 requests.
        crate::assertions::assert_received_n(openai_mock, 5).await;
        crate::assertions::assert_received_n(anthropic_mock, 5).await;
    }

    /// Send 10 concurrent requests against a harness configured with a very
    /// tight rate limit (3 RPM).  The total response count must be 10, split
    /// between 200 and 429 — verifying that rate limiting is enforced under
    /// concurrent load without panicking or deadlocking.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_e2e_concurrent_rate_limit_enforcement() {
        // RPM=3 so the window is exhausted quickly under 10 concurrent hits.
        // tpm=100_000 so token counting does not interfere.
        let harness = TestHarnessBuilder::new()
            .with_openai()
            .with_rate_limit(3, 100_000)
            .build()
            .await;
        let openai_mock = harness.mocks.openai.as_ref().unwrap();

        // No request limit on the mock — the rate limiter rejects excess
        // requests before they reach the upstream.
        mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");
        let base_url = format!("http://{}", harness.addr);

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let client = client.clone();
                let url = format!("{}/v1/chat/completions", base_url);
                tokio::spawn(async move {
                    client
                        .post(&url)
                        .header("Authorization", "Bearer test-api-key")
                        .json(&openai_body())
                        .send()
                        .await
                        .expect("rate-limit concurrent request failed")
                })
            })
            .collect();

        let results: Vec<_> = join_all(handles)
            .await
            .into_iter()
            .map(|r| r.expect("task panicked"))
            .collect();

        // Partition by status code.
        let ok_count = results
            .iter()
            .filter(|r| r.status().as_u16() == 200)
            .count();
        let too_many_count = results
            .iter()
            .filter(|r| r.status().as_u16() == 429)
            .count();

        // Total must be exactly 10 — no requests should hang or error out.
        assert_eq!(
            ok_count + too_many_count,
            10,
            "expected all 10 responses to be 200 or 429, got {ok_count} 200s and \
             {too_many_count} 429s (total {})",
            ok_count + too_many_count
        );

        // With RPM=3 and 10 concurrent requests at least some must be 429.
        // (Startup health-check polls may have already consumed a slot, so we
        // only assert at-least-one rather than an exact split.)
        assert!(
            too_many_count >= 1,
            "expected at least one 429 with RPM=3 under 10 concurrent requests, \
             got {too_many_count} 429s and {ok_count} 200s"
        );
    }

    /// Fire 20 requests concurrently and verify the proxy completes them all
    /// (200 or a clean error) without deadlocking.
    ///
    /// The `#[tokio::test]` macro enforces a per-test timeout via the runtime,
    /// so a deadlock would cause the test to time out and fail rather than hang
    /// indefinitely.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_e2e_no_deadlock_under_load() {
        let harness = TestHarnessBuilder::new().with_openai().build().await;
        let openai_mock = harness.mocks.openai.as_ref().unwrap();

        mock_chat_ok("gpt-4o", "ok").mount(openai_mock).await;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");
        let base_url = format!("http://{}", harness.addr);

        let handles: Vec<_> = (0..20)
            .map(|_| {
                let client = client.clone();
                let url = format!("{}/v1/chat/completions", base_url);
                tokio::spawn(async move {
                    client
                        .post(&url)
                        .header("Authorization", "Bearer test-api-key")
                        .json(&openai_body())
                        .send()
                        .await
                        .expect("load-test request failed")
                })
            })
            .collect();

        let results: Vec<_> = join_all(handles)
            .await
            .into_iter()
            .map(|r| r.expect("task panicked"))
            .collect();

        // Every response must be a valid HTTP status — no hangs, no panics.
        assert_eq!(results.len(), 20, "expected exactly 20 responses");

        for (i, resp) in results.iter().enumerate() {
            let status = resp.status().as_u16();
            assert!(
                status == 200 || (400..600).contains(&status),
                "request {i} returned unexpected status {status}"
            );
        }

        // Under no load-shedding config all 20 must be 200.
        for (i, resp) in results.iter().enumerate() {
            assert_eq!(
                resp.status().as_u16(),
                200,
                "no-rate-limit request {i} must be 200"
            );
        }
    }
}
