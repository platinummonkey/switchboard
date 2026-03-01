//! Integration tests for background key credential refresh.
//!
//! NOTE: spawn_refresh_task is not currently called by run_server.
//! These tests exercise the refresh loop in isolation, directly driving the
//! machinery without going through the full proxy stack.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use std::sync::RwLock;

use switchboard_server::auth::{AuthProvider, UpstreamAuthError, UpstreamCredentials};
use switchboard_server::key_pool::provider::PooledKey;
use switchboard_server::key_pool::spawn_refresh_task;

// ── Mock AuthProvider implementations ─────────────────────────────────────────

/// A provider that always returns fresh credentials and counts refresh calls.
struct AlwaysSucceedsProvider {
    name: &'static str,
    /// Counts calls to `refresh()`.
    refresh_count: Arc<AtomicUsize>,
    /// Counts calls to `get_credentials()`.
    get_count: Arc<AtomicUsize>,
    /// The header value returned by both methods.
    new_header_value: &'static str,
}

impl AlwaysSucceedsProvider {
    fn new(name: &'static str, new_header_value: &'static str) -> (Arc<Self>, Arc<AtomicUsize>) {
        let refresh_count = Arc::new(AtomicUsize::new(0));
        let get_count = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(Self {
            name,
            refresh_count: Arc::clone(&refresh_count),
            get_count,
            new_header_value,
        });
        (provider, refresh_count)
    }
}

#[async_trait]
impl AuthProvider for AlwaysSucceedsProvider {
    fn name(&self) -> &str {
        self.name
    }

    async fn get_credentials(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
        self.get_count.fetch_add(1, Ordering::SeqCst);
        Ok(UpstreamCredentials {
            header_name: HeaderName::from_static("authorization"),
            header_value: HeaderValue::from_str(self.new_header_value)
                .expect("invalid header value in test"),
            expires_at: Some(Instant::now() + Duration::from_secs(3600)),
        })
    }

    async fn refresh(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
        self.refresh_count.fetch_add(1, Ordering::SeqCst);
        self.get_credentials().await
    }

    fn is_valid(&self) -> bool {
        true
    }
}

/// A provider that succeeds only after a configurable number of failures.
struct FailsThenSucceedsProvider {
    name: &'static str,
    /// Number of consecutive failures before succeeding.
    fail_count: AtomicUsize,
    /// Remaining failures before returning success.
    failures_remaining: AtomicUsize,
    /// Total successful refresh calls.
    success_count: Arc<AtomicUsize>,
}

impl FailsThenSucceedsProvider {
    fn new(name: &'static str, failures: usize) -> (Arc<Self>, Arc<AtomicUsize>) {
        let success_count = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(Self {
            name,
            fail_count: AtomicUsize::new(0),
            failures_remaining: AtomicUsize::new(failures),
            success_count: Arc::clone(&success_count),
        });
        (provider, success_count)
    }
}

#[async_trait]
impl AuthProvider for FailsThenSucceedsProvider {
    fn name(&self) -> &str {
        self.name
    }

    async fn get_credentials(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
        // Not used in the refresh loop; delegate to refresh.
        self.refresh().await
    }

    async fn refresh(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
        // If failures remain, consume one and fail.
        let remaining = self.failures_remaining.load(Ordering::SeqCst);
        if remaining > 0 {
            self.failures_remaining.fetch_sub(1, Ordering::SeqCst);
            self.fail_count.fetch_add(1, Ordering::SeqCst);
            return Err(UpstreamAuthError::RefreshFailed("transient error".into()));
        }
        self.success_count.fetch_add(1, Ordering::SeqCst);
        Ok(UpstreamCredentials {
            header_name: HeaderName::from_static("authorization"),
            header_value: HeaderValue::from_static("Bearer refreshed-cred"),
            expires_at: Some(Instant::now() + Duration::from_secs(3600)),
        })
    }

    fn is_valid(&self) -> bool {
        true
    }
}

/// A provider that always fails to refresh.
struct AlwaysFailsProvider {
    call_count: Arc<AtomicUsize>,
}

#[async_trait]
impl AuthProvider for AlwaysFailsProvider {
    fn name(&self) -> &str {
        "always-fails"
    }

    async fn get_credentials(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Err(UpstreamAuthError::FetchFailed("unavailable".into()))
    }

    async fn refresh(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Err(UpstreamAuthError::RefreshFailed("network error".into()))
    }

    fn is_valid(&self) -> bool {
        false
    }
}

/// A provider whose `get_credentials` is never expected to be called.
/// Any call to `get_credentials` panics in debug mode via a counter check.
struct NeverCalledProvider {
    call_count: Arc<AtomicUsize>,
}

#[async_trait]
impl AuthProvider for NeverCalledProvider {
    fn name(&self) -> &str {
        "never-called"
    }

    async fn get_credentials(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        // Returning an error is fine; the point is we count the call.
        Err(UpstreamAuthError::FetchFailed(
            "should not be called".into(),
        ))
    }

    async fn refresh(&self) -> Result<UpstreamCredentials, UpstreamAuthError> {
        // refresh() is used by spawn_refresh_task, not get_credentials().
        // We track calls here too for completeness.
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Err(UpstreamAuthError::RefreshFailed(
            "should not be called".into(),
        ))
    }

    fn is_valid(&self) -> bool {
        true
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_key_with_expiry(id: &str, header_value: &'static str, expires_in: Duration) -> PooledKey {
    PooledKey::new_static(
        id,
        UpstreamCredentials {
            header_name: HeaderName::from_static("authorization"),
            header_value: HeaderValue::from_static(header_value),
            expires_at: Some(Instant::now() + expires_in),
        },
        1.0,
    )
}

fn make_key_no_expiry(id: &str, header_value: &'static str) -> PooledKey {
    PooledKey::new_static(
        id,
        UpstreamCredentials {
            header_name: HeaderName::from_static("authorization"),
            header_value: HeaderValue::from_static(header_value),
            expires_at: None,
        },
        1.0,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// spawn_refresh_task calls provider.refresh() and updates the key's credentials
/// before the credentials expire.
#[tokio::test]
async fn test_refresh_task_updates_credentials_before_expiry() {
    let (provider, refresh_count) = AlwaysSucceedsProvider::new("test-provider", "Bearer new-cred");

    // Key expires in 200 ms; buffer is 50 ms → refresh fires at ~150 ms.
    let key = Arc::new(RwLock::new(make_key_with_expiry(
        "k1",
        "Bearer old-cred",
        Duration::from_millis(200),
    )));

    let handle = spawn_refresh_task(
        Arc::clone(&key),
        provider as Arc<dyn AuthProvider>,
        Duration::from_millis(50),
    );

    // Wait long enough for the refresh to fire.
    tokio::time::sleep(Duration::from_millis(300)).await;
    handle.abort();

    assert!(
        refresh_count.load(Ordering::SeqCst) >= 1,
        "provider.refresh() should have been called at least once"
    );

    let guard = key.read().unwrap();
    let val = guard.credentials.header_value.to_str().unwrap();
    assert_eq!(
        val, "Bearer new-cred",
        "key credentials should be updated to the new value"
    );
}

/// Keys with no expiry (static keys) must never trigger a refresh call.
#[tokio::test]
async fn test_refresh_task_does_not_refresh_static_key() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn AuthProvider> = Arc::new(NeverCalledProvider {
        call_count: Arc::clone(&call_count),
    });

    // No expiry — the task should sleep for 1 hour and never call refresh.
    let key = Arc::new(RwLock::new(make_key_no_expiry("static-k", "Bearer static")));

    let handle = spawn_refresh_task(Arc::clone(&key), provider, Duration::from_secs(30));

    // Let the task run briefly; it should park without making any calls.
    tokio::time::sleep(Duration::from_millis(200)).await;
    handle.abort();

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        0,
        "provider should never be called for a key with no expiry"
    );

    // Credentials must remain unchanged.
    let guard = key.read().unwrap();
    let val = guard.credentials.header_value.to_str().unwrap();
    assert_eq!(val, "Bearer static");
}

/// On provider failure the task retries with exponential back-off and
/// eventually succeeds when the provider recovers.
///
/// The refresh task retries with INITIAL_RETRY_DELAY = 1 s.  This test uses
/// a provider that fails only once so the total wall-clock wait is ~1 s.
#[tokio::test(flavor = "multi_thread")]
async fn test_refresh_task_retries_on_failure() {
    // Provider fails once before succeeding on the second attempt.
    let (provider, success_count) = FailsThenSucceedsProvider::new("failing-provider", 1);

    // Key expires almost immediately so the refresh fires right away.
    let key = Arc::new(RwLock::new(make_key_with_expiry(
        "k-retry",
        "Bearer old-cred",
        Duration::from_millis(10),
    )));

    let handle = spawn_refresh_task(
        Arc::clone(&key),
        provider as Arc<dyn AuthProvider>,
        Duration::from_millis(5),
    );

    // First attempt fires ~immediately and fails.
    // The retry delay is 1 s, so wait 1.5 s to be safe.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    handle.abort();

    assert!(
        success_count.load(Ordering::SeqCst) >= 1,
        "provider should have succeeded at least once after retrying"
    );

    let guard = key.read().unwrap();
    let val = guard.credentials.header_value.to_str().unwrap();
    assert_eq!(
        val, "Bearer refreshed-cred",
        "credentials should be updated after eventual success"
    );
}

/// The task does not panic and the key is not modified when the provider
/// always fails — credentials remain the original value.
#[tokio::test]
async fn test_refresh_task_leaves_credentials_unchanged_on_persistent_failure() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn AuthProvider> = Arc::new(AlwaysFailsProvider {
        call_count: Arc::clone(&call_count),
    });

    // Key expires in 10 ms, buffer 5 ms → refresh fires almost immediately.
    let key = Arc::new(RwLock::new(make_key_with_expiry(
        "k-fail",
        "Bearer original",
        Duration::from_millis(10),
    )));

    let handle = spawn_refresh_task(Arc::clone(&key), provider, Duration::from_millis(5));

    // Allow the first attempt plus a brief wait; the retry delay is 1 s so
    // only one attempt should fire in 80 ms.
    tokio::time::sleep(Duration::from_millis(80)).await;
    handle.abort();

    // At least one refresh attempt was made.
    assert!(
        call_count.load(Ordering::SeqCst) >= 1,
        "provider should have been called at least once"
    );

    // The original credentials must be unchanged.
    let guard = key.read().unwrap();
    let val = guard.credentials.header_value.to_str().unwrap();
    assert_eq!(
        val, "Bearer original",
        "credentials must not change when every refresh fails"
    );
}

/// Two concurrent refresh tasks on different keys do not interfere.
#[tokio::test]
async fn test_two_concurrent_refresh_tasks_are_independent() {
    let (provider_a, refresh_count_a) = AlwaysSucceedsProvider::new("provider-a", "Bearer new-a");
    let (provider_b, refresh_count_b) = AlwaysSucceedsProvider::new("provider-b", "Bearer new-b");

    let key_a = Arc::new(RwLock::new(make_key_with_expiry(
        "ka",
        "Bearer old-a",
        Duration::from_millis(100),
    )));
    let key_b = Arc::new(RwLock::new(make_key_with_expiry(
        "kb",
        "Bearer old-b",
        Duration::from_millis(150),
    )));

    let handle_a = spawn_refresh_task(
        Arc::clone(&key_a),
        provider_a as Arc<dyn AuthProvider>,
        Duration::from_millis(20),
    );
    let handle_b = spawn_refresh_task(
        Arc::clone(&key_b),
        provider_b as Arc<dyn AuthProvider>,
        Duration::from_millis(20),
    );

    tokio::time::sleep(Duration::from_millis(300)).await;

    handle_a.abort();
    handle_b.abort();

    assert!(
        refresh_count_a.load(Ordering::SeqCst) >= 1,
        "key_a should have been refreshed"
    );
    assert!(
        refresh_count_b.load(Ordering::SeqCst) >= 1,
        "key_b should have been refreshed"
    );

    let guard_a = key_a.read().unwrap();
    assert_eq!(
        guard_a.credentials.header_value.to_str().unwrap(),
        "Bearer new-a"
    );

    let guard_b = key_b.read().unwrap();
    assert_eq!(
        guard_b.credentials.header_value.to_str().unwrap(),
        "Bearer new-b"
    );
}

/// A key that expires very soon (within the buffer) triggers an immediate
/// refresh rather than sleeping first.
#[tokio::test]
async fn test_refresh_task_fires_immediately_when_already_past_refresh_deadline() {
    let (provider, refresh_count) =
        AlwaysSucceedsProvider::new("immediate-provider", "Bearer immediate-new");

    // Key expires in 5 ms but buffer is 50 ms → refresh_at is clamped to now,
    // so the task should refresh immediately without sleeping.
    let key = Arc::new(RwLock::new(make_key_with_expiry(
        "k-immediate",
        "Bearer immediate-old",
        Duration::from_millis(5),
    )));

    let handle = spawn_refresh_task(
        Arc::clone(&key),
        provider as Arc<dyn AuthProvider>,
        Duration::from_millis(50), // buffer > expiry → deadline is now
    );

    // Should be fast — 100 ms is plenty.
    tokio::time::sleep(Duration::from_millis(100)).await;
    handle.abort();

    assert!(
        refresh_count.load(Ordering::SeqCst) >= 1,
        "refresh should fire immediately when already past the refresh deadline"
    );

    let guard = key.read().unwrap();
    assert_eq!(
        guard.credentials.header_value.to_str().unwrap(),
        "Bearer immediate-new"
    );
}
