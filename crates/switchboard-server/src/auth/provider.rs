//! Pluggable auth provider trait for server→upstream credential injection.

use std::time::Instant;

use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use thiserror::Error;

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum UpstreamAuthError {
    #[error("credential fetch failed: {0}")]
    FetchFailed(String),

    #[error("credential refresh failed: {0}")]
    RefreshFailed(String),

    #[error("credentials expired and could not be refreshed")]
    Expired,

    #[error("provider misconfigured: {0}")]
    Config(String),
}

// ── Upstream credentials ──────────────────────────────────────────────────────

/// A single HTTP header to inject into upstream LLM provider requests.
/// Produced by [`AuthProvider::get_credentials`].
#[derive(Clone)]
pub struct UpstreamCredentials {
    /// Header name, e.g. `Authorization` or `x-api-key`.
    pub header_name: HeaderName,
    /// Header value, e.g. `Bearer sk-…`.
    pub header_value: HeaderValue,
    /// When these credentials expire; `None` means they do not expire.
    pub expires_at: Option<Instant>,
}

impl UpstreamCredentials {
    /// Returns `true` if the credentials have a known expiry and it has passed.
    pub fn is_expired(&self) -> bool {
        self.expires_at.is_some_and(|t| Instant::now() >= t)
    }

    /// Returns `true` if the credentials are still usable.
    pub fn is_valid(&self) -> bool {
        !self.is_expired()
    }
}

impl std::fmt::Debug for UpstreamCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpstreamCredentials")
            .field("header_name", &self.header_name)
            .field("header_value", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

// ── AuthProvider trait ────────────────────────────────────────────────────────

/// Pluggable provider that supplies upstream LLM credentials.
///
/// Each instance represents one credential slot (e.g., one Anthropic API key).
/// The implementation is responsible for caching and proactive refresh.
///
/// Implementors: `StaticKeyProvider`, `JwtProvider`, `AwsStsProvider`,
/// `VaultProvider` (Phases 4–5).
#[async_trait]
pub trait AuthProvider: Send + Sync + 'static {
    /// Short unique name used in logs and metrics, e.g. `"anthropic-prod-1"`.
    fn name(&self) -> &str;

    /// Return currently-valid credentials, refreshing if necessary.
    async fn get_credentials(&self) -> Result<UpstreamCredentials, UpstreamAuthError>;

    /// Force-refresh credentials (called after a 401 from upstream).
    async fn refresh(&self) -> Result<UpstreamCredentials, UpstreamAuthError>;

    /// Fast, synchronous validity check — no I/O.
    fn is_valid(&self) -> bool;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// Verify `AuthProvider` is object-safe by constructing a `dyn` reference.
    fn _assert_object_safe(_: &dyn AuthProvider) {}

    /// A minimal no-op implementation used only in compile/trait tests.
    struct ConstantProvider {
        name: &'static str,
        creds: UpstreamCredentials,
    }

    #[async_trait]
    impl AuthProvider for ConstantProvider {
        fn name(&self) -> &str {
            self.name
        }
        async fn get_credentials(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
            Ok(self.creds.clone())
        }
        async fn refresh(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
            Ok(self.creds.clone())
        }
        fn is_valid(&self) -> bool {
            true
        }
    }

    fn make_creds(expires: Option<Instant>) -> UpstreamCredentials {
        UpstreamCredentials {
            header_name: HeaderName::from_static("authorization"),
            header_value: HeaderValue::from_static("Bearer sk-test"),
            expires_at: expires,
        }
    }

    #[test]
    fn test_creds_no_expiry_are_valid() {
        let c = make_creds(None);
        assert!(c.is_valid());
        assert!(!c.is_expired());
    }

    #[test]
    fn test_creds_future_expiry_are_valid() {
        let c = make_creds(Some(Instant::now() + std::time::Duration::from_secs(3600)));
        assert!(c.is_valid());
    }

    #[test]
    fn test_creds_past_expiry_are_expired() {
        let c = make_creds(Some(Instant::now() - std::time::Duration::from_secs(1)));
        assert!(c.is_expired());
        assert!(!c.is_valid());
    }

    #[test]
    fn test_creds_debug_redacts_value() {
        let c = make_creds(None);
        let s = format!("{c:?}");
        assert!(s.contains("<redacted>"));
        assert!(!s.contains("sk-test"));
    }

    #[tokio::test]
    async fn test_provider_impl_compiles_and_runs() {
        let p = Arc::new(ConstantProvider {
            name: "test",
            creds: make_creds(None),
        });
        assert_eq!(p.name(), "test");
        assert!(p.is_valid());
        let creds = p.get_credentials().await.unwrap();
        assert_eq!(creds.header_name, "authorization");
        let refreshed = p.refresh().await.unwrap();
        assert!(refreshed.is_valid());
    }

    #[test]
    fn test_auth_error_display() {
        let e = UpstreamAuthError::FetchFailed("timeout".into());
        assert!(e.to_string().contains("timeout"));
        let e2 = UpstreamAuthError::Expired;
        assert!(e2.to_string().contains("expired"));
    }
}
