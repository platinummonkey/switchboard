//! HashiCorp Vault KV v2 key provider.
//!
//! Reads an API key from a Vault KV v2 secret path and returns it as
//! `Authorization: Bearer <api_key>` upstream credentials.

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

        tracing::info!(path = %self.path, "successfully fetched key from Vault");

        Ok(UpstreamCredentials {
            header_name: HeaderName::from_static("authorization"),
            header_value,
            // Vault secrets do not carry an expiry in this path; rotation is
            // handled externally via lease renewal or periodic re-fetch.
            expires_at: None,
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
}
