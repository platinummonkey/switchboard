//! HTTP test client for E2E tests.
//!
//! Wraps [`reqwest::Client`] with helper methods for each switchboard
//! endpoint, pre-injecting the test API key and base URLs.

use std::net::SocketAddr;

use reqwest::Response;
use serde_json::Value;

/// Pre-configured HTTP client pointing at a running switchboard test server.
pub struct TestClient {
    inner: reqwest::Client,
    base_url: String,
    admin_url: Option<String>,
    api_key: String,
}

impl TestClient {
    /// Create a new test client targeting `proxy_addr`.
    ///
    /// `admin_addr` is `Some` only when the harness was built with
    /// `.with_admin()`. Calling `admin_*` methods on a client without an
    /// admin URL will panic with a clear message.
    pub fn new(proxy_addr: SocketAddr, admin_addr: Option<SocketAddr>) -> Self {
        Self {
            inner: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build reqwest client"),
            base_url: format!("http://{}", proxy_addr),
            admin_url: admin_addr.map(|a| format!("http://{}", a)),
            api_key: "test-api-key".into(),
        }
    }

    // ── Proxy endpoints ────────────────────────────────────────────────────────

    /// `POST /v1/chat/completions` with the test API key.
    pub async fn chat_completions(&self, body: Value) -> Response {
        self.inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .expect("chat_completions request failed")
    }

    /// `POST /v1/chat/completions` with `stream: true` injected.
    pub async fn chat_completions_stream(&self, mut body: Value) -> Response {
        body["stream"] = serde_json::json!(true);
        self.inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .expect("chat_completions_stream request failed")
    }

    /// `POST /api/v1/messages` (Anthropic-compatible endpoint) with test API key.
    pub async fn anthropic_messages(&self, body: Value) -> Response {
        self.inner
            .post(format!("{}/api/v1/messages", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .expect("anthropic_messages request failed")
    }

    /// `POST /api/v1/messages` with `stream: true` injected.
    pub async fn anthropic_messages_stream(&self, mut body: Value) -> Response {
        body["stream"] = serde_json::json!(true);
        self.inner
            .post(format!("{}/api/v1/messages", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .expect("anthropic_messages_stream request failed")
    }

    /// `POST /v1/chat/completions` with the test API key plus extra headers.
    ///
    /// Used by identity-resolution tests that need to inject
    /// `x-switchboard-user`, `x-switchboard-tool`, or JWT-claim headers.
    pub async fn chat_with_extra_headers(&self, body: Value, extra: &[(&str, &str)]) -> Response {
        let mut req = self
            .inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body);
        for (name, value) in extra {
            req = req.header(*name, *value);
        }
        req.send()
            .await
            .expect("chat_with_extra_headers request failed")
    }

    /// `POST /v1/chat/completions` WITHOUT any Authorization header.
    ///
    /// Used to test that unauthenticated requests are rejected with 401.
    pub async fn chat_no_auth(&self, body: Value) -> Response {
        self.inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&body)
            .send()
            .await
            .expect("chat_no_auth request failed")
    }

    /// `POST /v1/chat/completions` with a specific (possibly invalid) API key.
    pub async fn chat_with_key(&self, body: Value, api_key: &str) -> Response {
        self.inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await
            .expect("chat_with_key request failed")
    }

    /// `GET /health` liveness probe.
    ///
    /// Includes the test API key header so the auth layer lets the request
    /// through and the server returns 200 OK when healthy.
    pub async fn health(&self) -> Response {
        self.inner
            .get(format!("{}/health", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .expect("health request failed")
    }

    // ── Admin endpoints ────────────────────────────────────────────────────────

    fn admin_base(&self) -> &str {
        self.admin_url
            .as_deref()
            .expect("admin not enabled in this TestHarness; use .with_admin() on the builder")
    }

    /// `GET /admin/api/v1/{path}` with the test API key.
    pub async fn admin_get(&self, path: &str) -> Response {
        self.inner
            .get(format!("{}/admin/api/v1/{}", self.admin_base(), path))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .expect("admin_get request failed")
    }

    /// `PUT /admin/api/v1/{path}` with the test API key and a JSON body.
    pub async fn admin_put(&self, path: &str, body: Value) -> Response {
        self.inner
            .put(format!("{}/admin/api/v1/{}", self.admin_base(), path))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .expect("admin_put request failed")
    }

    /// `DELETE /admin/api/v1/{path}` with the test API key.
    pub async fn admin_delete(&self, path: &str) -> Response {
        self.inner
            .delete(format!("{}/admin/api/v1/{}", self.admin_base(), path))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .expect("admin_delete request failed")
    }

    /// `POST /admin/api/v1/{path}` with the test API key and a JSON body.
    pub async fn admin_post(&self, path: &str, body: Value) -> Response {
        self.inner
            .post(format!("{}/admin/api/v1/{}", self.admin_base(), path))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .expect("admin_post request failed")
    }

    // ── SSE helpers ────────────────────────────────────────────────────────────

    /// Consume a streaming response and collect all `data:` payloads.
    ///
    /// Strips the `data: ` prefix and filters out `[DONE]` sentinels.
    /// Returns the raw payload strings in order.
    pub async fn collect_sse(resp: Response) -> Vec<String> {
        let bytes = resp.bytes().await.expect("failed to read SSE body");
        let text = std::str::from_utf8(&bytes).expect("SSE body is not UTF-8");
        text.lines()
            .filter(|l| l.starts_with("data: ") && *l != "data: [DONE]")
            .map(|l| l.trim_start_matches("data: ").to_string())
            .collect()
    }
}
