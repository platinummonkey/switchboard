//! HashiCorp Vault KV v2 key provider.
//!
//! Reads an API key from a Vault KV v2 secret path and returns it as
//! `Authorization: Bearer <api_key>` upstream credentials.
//!
//! TTL handling:
//! - Dynamic secrets carry `lease_duration` (seconds) in the top-level
//!   response.  When non-zero this is used to compute `expires_at`.
//! - KV v2 secrets may carry a `data.metadata.deletion_time` RFC-3339
//!   timestamp.  When present and non-empty this is parsed and used instead.
//! - A 30-second buffer is subtracted so that the refresh fires before the
//!   credential actually expires.

use async_trait::async_trait;
use http::{HeaderName, HeaderValue};

use crate::auth::UpstreamCredentials;
use crate::error::ServerError;
use crate::key_pool::provider::KeyProvider;

// ── VaultProvider ─────────────────────────────────────────────────────────────

/// Reads a secret from HashiCorp Vault KV v2 and returns it as an
/// `Authorization: Bearer` credential.
///
/// Expected secret structure at `{vault_addr}/v1/{path}`:
/// ```json
/// {"data": {"data": {"api_key": "sk-..."}}}
/// ```
#[derive(Debug, Clone)]
pub struct VaultProvider {
    /// KV v2 path, e.g. `secret/data/openai/prod-key`.
    path: String,
    /// Vault server address, e.g. `https://vault.example.com:8200`.
    vault_addr: String,
    /// Vault token used for authentication.
    token: String,
}

impl VaultProvider {
    /// Create a new Vault provider.
    ///
    /// # Arguments
    /// * `path`       — KV v2 secret path (without `/v1/` prefix).
    /// * `vault_addr` — Base URL of the Vault server.
    /// * `token`      — Vault authentication token.
    pub fn new(
        path: impl Into<String>,
        vault_addr: impl Into<String>,
        token: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            vault_addr: vault_addr.into(),
            token: token.into(),
        }
    }
}

#[async_trait]
impl KeyProvider for VaultProvider {
    async fn fetch(&self) -> Result<UpstreamCredentials, ServerError> {
        let url = format!("{}/v1/{}", self.vault_addr.trim_end_matches('/'), self.path);

        tracing::debug!(path = %self.path, vault_addr = %self.vault_addr, "reading secret from Vault");

        let client = reqwest::Client::new();
        let resp = client
            .get(&url)
            .header("X-Vault-Token", &self.token)
            .send()
            .await
            .map_err(|e| ServerError::Config(format!("Vault read failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            tracing::warn!(path = %self.path, status = %status, "Vault returned non-success status");
            return Err(ServerError::Config(format!("Vault returned {status}")));
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| ServerError::Config(format!("Vault JSON parse failed: {e}")))?;

        let api_key = body["data"]["data"]["api_key"]
            .as_str()
            .ok_or_else(|| ServerError::Config("Vault: missing api_key field".into()))?;

        let header_value = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|e| ServerError::Config(format!("invalid Vault header value: {e}")))?;

        // ── TTL extraction ────────────────────────────────────────────────────
        // Dynamic secrets: top-level `lease_duration` (seconds, non-zero).
        let lease_secs = body
            .get("lease_duration")
            .and_then(|v| v.as_u64())
            .filter(|&d| d > 0);

        // KV v2 secrets: `data.metadata.deletion_time` (RFC-3339, non-empty).
        let deletion_time_secs = body
            .pointer("/data/metadata/deletion_time")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| {
                let now = chrono::Utc::now();
                let remaining = dt.signed_duration_since(now).num_seconds();
                remaining.max(0) as u64
            });

        // Prefer lease_duration (dynamic) over deletion_time (KV v2).
        let ttl_secs = lease_secs.or(deletion_time_secs);

        // Subtract a 30-second buffer so we refresh before the actual expiry.
        let expires_at = ttl_secs.map(|secs| {
            let effective = secs.saturating_sub(30);
            std::time::Instant::now() + std::time::Duration::from_secs(effective)
        });

        if let Some(ref exp) = expires_at {
            tracing::info!(
                path = %self.path,
                ttl_secs = ttl_secs.unwrap_or(0),
                expires_in_secs = exp.duration_since(std::time::Instant::now()).as_secs(),
                "successfully fetched key from Vault with TTL"
            );
        } else {
            tracing::info!(path = %self.path, "successfully fetched key from Vault (no TTL)");
        }

        Ok(UpstreamCredentials {
            header_name: HeaderName::from_static("authorization"),
            header_value,
            expires_at,
        })
    }

    fn description(&self) -> &str {
        &self.path
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn make_provider(vault_addr: &str, secret_path: &str) -> VaultProvider {
        VaultProvider::new(secret_path, vault_addr, "s.testtoken")
    }

    #[test]
    fn test_vault_provider_description() {
        let p = make_provider("https://vault.example.com", "secret/data/openai");
        assert_eq!(p.description(), "secret/data/openai");
    }

    #[tokio::test]
    async fn test_vault_provider_fetch_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/secret/data/openai"))
            .and(header("X-Vault-Token", "s.testtoken"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "data": {
                        "api_key": "sk-test-vault-key"
                    },
                    "metadata": {"version": 1}
                }
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri(), "secret/data/openai");
        let creds = provider.fetch().await.unwrap();

        let val = creds.header_value.to_str().unwrap();
        assert_eq!(val, "Bearer sk-test-vault-key");
        assert_eq!(creds.header_name, "authorization");
        assert!(creds.expires_at.is_none());
    }

    #[tokio::test]
    async fn test_vault_provider_fetch_404() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/secret/data/missing"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "errors": []
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri(), "secret/data/missing");
        let err = provider.fetch().await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("404"),
            "error message should mention 404, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_vault_provider_fetch_missing_key() {
        let mock_server = MockServer::start().await;

        // Response with the `api_key` field absent.
        Mock::given(method("GET"))
            .and(path("/v1/secret/data/no-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "data": {}
                }
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri(), "secret/data/no-key");
        let err = provider.fetch().await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("api_key"),
            "error message should mention api_key, got: {msg}"
        );
    }

    /// Dynamic-secret response with `lease_duration: 3600` → `expires_at`
    /// should be `Some(...)` set to roughly 3570 seconds from now (3600 - 30
    /// buffer).
    #[tokio::test]
    async fn test_vault_fetch_parses_lease_duration() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/db/creds/my-role"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "lease_id": "db/creds/my-role/abc123",
                "lease_duration": 3600,
                "renewable": true,
                "data": {
                    "data": { "api_key": "sk-test" }
                }
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri(), "db/creds/my-role");
        let before = std::time::Instant::now();
        let creds = provider.fetch().await.unwrap();
        let after = std::time::Instant::now();

        let expires_at = creds
            .expires_at
            .expect("expires_at should be Some for lease_duration=3600");

        // effective TTL = 3600 - 30 = 3570 seconds
        // expires_at must be between (before + 3569 s) and (after + 3571 s).
        let lower = before + std::time::Duration::from_secs(3569);
        let upper = after + std::time::Duration::from_secs(3571);
        assert!(
            expires_at >= lower && expires_at <= upper,
            "expires_at out of expected range"
        );

        let val = creds.header_value.to_str().unwrap();
        assert_eq!(val, "Bearer sk-test");
    }

    /// KV v2 response with `lease_duration: 0` and no `deletion_time` →
    /// `expires_at` should be `None`.
    #[tokio::test]
    async fn test_vault_fetch_no_ttl_returns_none_expiry() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/secret/data/static-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "lease_id": "",
                "lease_duration": 0,
                "renewable": false,
                "data": {
                    "data": { "api_key": "sk-static" },
                    "metadata": {
                        "created_time": "2024-01-01T00:00:00Z",
                        "deletion_time": "",
                        "destroyed": false,
                        "version": 1
                    }
                }
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri(), "secret/data/static-key");
        let creds = provider.fetch().await.unwrap();

        assert!(
            creds.expires_at.is_none(),
            "lease_duration=0 and empty deletion_time should yield expires_at=None"
        );
    }

    /// KV v2 response with a `deletion_time` far in the future → `expires_at`
    /// should be `Some(...)` far in the future.
    #[tokio::test]
    async fn test_vault_fetch_parses_deletion_time() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/secret/data/expiring-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "lease_id": "",
                "lease_duration": 0,
                "renewable": false,
                "data": {
                    "data": { "api_key": "sk-expiring" },
                    "metadata": {
                        "created_time": "2024-01-01T00:00:00Z",
                        "deletion_time": "2099-01-01T00:00:00Z",
                        "destroyed": false,
                        "version": 1
                    }
                }
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri(), "secret/data/expiring-key");
        let creds = provider.fetch().await.unwrap();

        let expires_at = creds
            .expires_at
            .expect("deletion_time in 2099 should yield Some(expires_at)");

        // The remaining TTL should be many years (>> 1 year of seconds).
        let remaining = expires_at.duration_since(std::time::Instant::now());
        let one_year_secs = 365u64 * 24 * 3600;
        assert!(
            remaining.as_secs() > one_year_secs,
            "expires_at should be far in the future, got {} secs remaining",
            remaining.as_secs()
        );
    }

    /// KV v2 response with a `deletion_time` in the past → the remaining TTL
    /// is 0 after clamping, so after subtracting the 30-second buffer the
    /// effective duration is 0.  `expires_at` should be `Some(Instant::now())`
    /// (i.e. immediately expired).
    #[tokio::test]
    async fn test_vault_fetch_expired_deletion_time() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/secret/data/past-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "lease_id": "",
                "lease_duration": 0,
                "renewable": false,
                "data": {
                    "data": { "api_key": "sk-past" },
                    "metadata": {
                        "created_time": "2020-01-01T00:00:00Z",
                        "deletion_time": "2020-06-01T00:00:00Z",
                        "destroyed": false,
                        "version": 1
                    }
                }
            })))
            .mount(&mock_server)
            .await;

        let provider = make_provider(&mock_server.uri(), "secret/data/past-key");
        let before = std::time::Instant::now();
        let creds = provider.fetch().await.unwrap();

        let expires_at = creds
            .expires_at
            .expect("past deletion_time should still yield Some(expires_at)");

        // remaining = 0 (clamped), effective = 0.saturating_sub(30) = 0
        // So expires_at ≈ Instant::now() at fetch time — must be <= a generous
        // upper bound a few seconds after `before`.
        let upper = before + std::time::Duration::from_secs(5);
        assert!(
            expires_at <= upper,
            "expired deletion_time should produce an immediately-expired expires_at"
        );
    }
}
