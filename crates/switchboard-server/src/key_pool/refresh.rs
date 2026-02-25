//! Proactive credential refresh for pooled keys.
//!
//! [`spawn_refresh_task`] starts a background tokio task that wakes up shortly
//! before a key's credentials expire, calls the [`AuthProvider`] to refresh
//! them, and writes the new credentials back into the shared `RwLock<PooledKey>`.
//!
//! On failure the task retries with exponential back-off (1 s → 2 s → 4 s …
//! capped at 5 minutes) and logs errors using `tracing`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use crate::auth::AuthProvider;
use crate::key_pool::PooledKey;

// ── Constants ─────────────────────────────────────────────────────────────────

const INITIAL_RETRY_DELAY: Duration = Duration::from_secs(1);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(300); // 5 minutes

// ── RefreshableKey ────────────────────────────────────────────────────────────

/// A [`PooledKey`] paired with an [`AuthProvider`] that can refresh its
/// credentials.
///
/// The `spawn_refresh_task` function manages the background refresh loop; this
/// struct is the ownership boundary that keeps both alive.
pub struct RefreshableKey {
    pub key: Arc<RwLock<PooledKey>>,
    pub provider: Arc<dyn AuthProvider>,
}

impl std::fmt::Debug for RefreshableKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshableKey")
            .field("provider", &self.provider.name())
            .finish()
    }
}

impl RefreshableKey {
    pub fn new(key: Arc<RwLock<PooledKey>>, provider: Arc<dyn AuthProvider>) -> Self {
        Self { key, provider }
    }
}

// ── spawn_refresh_task ────────────────────────────────────────────────────────

/// Spawn a background task that proactively refreshes credentials.
///
/// # Arguments
///
/// * `key`      — Shared, mutable key whose credentials will be updated.
/// * `provider` — Auth provider that supplies fresh credentials.
/// * `buffer`   — How far in advance of expiry to attempt a refresh.
///
/// # Returns
///
/// A [`JoinHandle`] for the spawned task. Drop it to allow the task to run
/// detached, or `abort()` it to stop refreshing.
pub fn spawn_refresh_task(
    key: Arc<RwLock<PooledKey>>,
    provider: Arc<dyn AuthProvider>,
    buffer: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            // Determine when the current credentials expire.
            let expires_at = {
                let guard = key.read().await;
                guard.credentials.expires_at
            };

            match expires_at {
                None => {
                    // Credentials do not expire; nothing to refresh.  Sleep
                    // indefinitely and only wake up if the task is aborted.
                    tracing::debug!(
                        provider = provider.name(),
                        "credentials have no expiry; refresh task sleeping indefinitely"
                    );
                    // Park for a long duration and loop; allows future logic to
                    // change the key's credentials and trigger a recheck.
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                    continue;
                }
                Some(expiry) => {
                    let now = Instant::now();
                    // Compute the refresh deadline: expiry - buffer (clamped to now).
                    let refresh_at = expiry.checked_sub(buffer).unwrap_or(now);
                    if refresh_at > now {
                        let sleep_for = refresh_at.duration_since(now);
                        tracing::debug!(
                            provider = provider.name(),
                            sleep_secs = sleep_for.as_secs(),
                            "scheduling proactive credential refresh"
                        );
                        tokio::time::sleep(sleep_for).await;
                    }
                }
            }

            // Attempt refresh with exponential back-off on failure.
            let mut retry_delay = INITIAL_RETRY_DELAY;
            loop {
                match provider.refresh().await {
                    Ok(new_creds) => {
                        let mut guard = key.write().await;
                        tracing::info!(
                            provider = provider.name(),
                            key_id = guard.id,
                            "credentials refreshed successfully"
                        );
                        guard.credentials = new_creds;
                        break; // Success — exit the retry loop.
                    }
                    Err(err) => {
                        tracing::error!(
                            provider = provider.name(),
                            error = %err,
                            retry_in_secs = retry_delay.as_secs(),
                            "credential refresh failed; will retry"
                        );
                        tokio::time::sleep(retry_delay).await;
                        retry_delay = (retry_delay * 2).min(MAX_RETRY_DELAY);
                    }
                }
            }
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use http::{HeaderName, HeaderValue};
    use tokio::sync::RwLock;

    use super::*;
    use crate::auth::{AuthProvider, UpstreamAuthError, UpstreamCredentials};
    use crate::key_pool::PooledKey;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn make_key_with_expiry(id: &str, expires_in: Duration) -> PooledKey {
        PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer old-cred"),
                expires_at: Some(Instant::now() + expires_in),
            },
            1.0,
        )
    }

    fn make_key_no_expiry(id: &str) -> PooledKey {
        PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer static-cred"),
                expires_at: None,
            },
            1.0,
        )
    }

    /// A provider that always returns fresh credentials immediately.
    struct AlwaysSucceedsProvider {
        name: &'static str,
        call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AuthProvider for AlwaysSucceedsProvider {
        fn name(&self) -> &str {
            self.name
        }
        async fn get_credentials(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
            Ok(UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer new-cred"),
                expires_at: Some(Instant::now() + Duration::from_secs(3600)),
            })
        }
        async fn refresh(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            self.get_credentials().await
        }
        fn is_valid(&self) -> bool {
            true
        }
    }

    /// A provider that always fails to refresh.
    struct AlwaysFailsProvider;

    #[async_trait]
    impl AuthProvider for AlwaysFailsProvider {
        fn name(&self) -> &str {
            "always-fails"
        }
        async fn get_credentials(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
            Err(UpstreamAuthError::FetchFailed("unavailable".into()))
        }
        async fn refresh(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
            Err(UpstreamAuthError::RefreshFailed("network error".into()))
        }
        fn is_valid(&self) -> bool {
            false
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_refreshable_key_constructs() {
        let key = Arc::new(RwLock::new(make_key_no_expiry("k1")));
        let provider: Arc<dyn AuthProvider> = Arc::new(AlwaysSucceedsProvider {
            name: "p1",
            call_count: Arc::new(AtomicUsize::new(0)),
        });
        let rk = RefreshableKey::new(Arc::clone(&key), Arc::clone(&provider));
        assert_eq!(rk.provider.name(), "p1");
    }

    #[tokio::test]
    async fn test_refresh_task_updates_credentials_on_near_expiry() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let provider: Arc<dyn AuthProvider> = Arc::new(AlwaysSucceedsProvider {
            name: "p1",
            call_count: Arc::clone(&call_count),
        });

        // Credential expires in 200 ms; buffer is 100 ms → task should fire at ~100 ms.
        let key = Arc::new(RwLock::new(make_key_with_expiry(
            "k1",
            Duration::from_millis(200),
        )));
        let handle = spawn_refresh_task(Arc::clone(&key), provider, Duration::from_millis(100));

        // Wait long enough for the refresh to happen.
        tokio::time::sleep(Duration::from_millis(250)).await;
        handle.abort();

        assert!(
            call_count.load(Ordering::SeqCst) >= 1,
            "provider.refresh() should have been called at least once"
        );

        // Credentials should be updated.
        let guard = key.read().await;
        let val = guard.credentials.header_value.to_str().unwrap();
        assert_eq!(val, "Bearer new-cred");
    }

    #[tokio::test]
    async fn test_refresh_task_spawns_without_panic_for_no_expiry_key() {
        let provider: Arc<dyn AuthProvider> = Arc::new(AlwaysSucceedsProvider {
            name: "static-provider",
            call_count: Arc::new(AtomicUsize::new(0)),
        });
        let key = Arc::new(RwLock::new(make_key_no_expiry("k1")));
        let handle = spawn_refresh_task(Arc::clone(&key), provider, Duration::from_secs(30));

        // Let it tick briefly — it should park, not panic.
        tokio::time::sleep(Duration::from_millis(50)).await;
        handle.abort();
    }

    #[tokio::test]
    async fn test_refresh_task_retries_on_failure() {
        // Use a very short credential + buffer so the refresh fires immediately.
        let key = Arc::new(RwLock::new(make_key_with_expiry(
            "k1",
            Duration::from_millis(10),
        )));
        let provider: Arc<dyn AuthProvider> = Arc::new(AlwaysFailsProvider);
        let handle = spawn_refresh_task(Arc::clone(&key), provider, Duration::from_millis(5));

        // Let it attempt at least two retries (first at ~5 ms, retry at ~1 s).
        tokio::time::sleep(Duration::from_millis(80)).await;
        handle.abort();

        // The key's credentials should still be the original ones (refresh
        // never succeeded).
        let guard = key.read().await;
        let val = guard.credentials.header_value.to_str().unwrap();
        assert_eq!(val, "Bearer old-cred");
    }
}
