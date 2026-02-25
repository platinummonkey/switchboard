//! `KeySelector` trait and `KeyPool` full implementation.
//!
//! Concrete selector implementations (WeightedRandom, RoundRobin, etc.)
//! live in Phase 5 (`key_pool/selector/`).

use std::time::{Duration, Instant};

use switchboard_common::types::RequestContext;

use crate::key_pool::PooledKey;
use crate::key_pool::health::KeyStatus;

// ── Selector trait ────────────────────────────────────────────────────────────

/// Chooses a key from the pool for a given request.
///
/// Implementations must be `Send + Sync` so they can be shared across async
/// tasks. Selection must be synchronous and fast (no I/O).
pub trait KeySelector: Send + Sync {
    /// Return a reference to the chosen key, or `None` if the pool is empty
    /// or all keys are disabled.
    fn select<'a>(&self, pool: &'a [PooledKey], request: &RequestContext) -> Option<&'a PooledKey>;
}

// ── Error threshold constants ─────────────────────────────────────────────────

/// Number of errors in the last 5 minutes that triggers the Degraded status.
const DEGRADED_THRESHOLD: u64 = 5;

// ── KeyPool ───────────────────────────────────────────────────────────────────

/// Runtime key pool for one upstream provider.
///
/// Holds an ordered list of [`PooledKey`] entries and delegates selection to
/// a pluggable [`KeySelector`].  Health updates and rotation algorithms are
/// updated at runtime via the health recording methods.
pub struct KeyPool {
    pub(crate) keys: Vec<PooledKey>,
    pub(crate) selector: Box<dyn KeySelector>,
}

impl KeyPool {
    /// Construct a pool from a pre-built key list and selector.
    pub fn new(keys: Vec<PooledKey>, selector: Box<dyn KeySelector>) -> Self {
        Self { keys, selector }
    }

    /// Number of keys in the pool (all statuses).
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// `true` if the pool contains no keys at all.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Number of keys currently eligible to serve requests.
    pub fn eligible_count(&self) -> usize {
        self.keys.iter().filter(|k| k.is_eligible()).count()
    }

    /// Select a key for the given request context.
    pub fn select(&self, request: &RequestContext) -> Option<&PooledKey> {
        self.selector.select(&self.keys, request)
    }

    // ── Mutation: adding / removing keys ──────────────────────────────────────

    /// Add a key at runtime (e.g. from the admin API).
    ///
    /// If a key with the same `id` already exists it is replaced.
    pub fn add_key(&mut self, key: PooledKey) {
        if let Some(existing) = self.keys.iter_mut().find(|k| k.id == key.id) {
            *existing = key;
        } else {
            self.keys.push(key);
        }
    }

    /// Remove a key by id.  Returns `true` if a key was found and removed.
    pub fn remove_key(&mut self, id: &str) -> bool {
        if let Some(pos) = self.keys.iter().position(|k| k.id == id) {
            self.keys.remove(pos);
            true
        } else {
            false
        }
    }

    /// Mutable access to a key by id, for health updates.
    pub fn get_key_mut(&mut self, id: &str) -> Option<&mut PooledKey> {
        self.keys.iter_mut().find(|k| k.id == id)
    }

    // ── Health recording ──────────────────────────────────────────────────────

    /// Record a successful upstream request.
    ///
    /// Increments `total_requests`, updates the exponential moving average of
    /// latency, and records the last-used timestamp.
    pub fn record_success(&mut self, key_id: &str, latency: Duration) {
        let Some(key) = self.get_key_mut(key_id) else {
            tracing::warn!(key_id, "record_success: key not found");
            return;
        };

        key.health.total_requests += 1;
        key.health.last_used = Some(Instant::now());

        // Exponential moving average: EMA = alpha * sample + (1-alpha) * ema
        // Use alpha = 0.1 so recent latency is smoothed over ~10 requests.
        let latency_ms = latency.as_secs_f64() * 1000.0;
        if key.health.total_requests == 1 {
            key.health.avg_latency_ms = latency_ms;
        } else {
            key.health.avg_latency_ms = 0.1 * latency_ms + 0.9 * key.health.avg_latency_ms;
        }
    }

    /// Record a failed upstream request.
    ///
    /// Increments `errors_last_5m` (or `rate_limit_hits_last_5m` when
    /// `is_rate_limit` is `true`) and applies status transition rules:
    ///
    /// - `errors_last_5m > 5`              → `Degraded`
    /// - `rate_limit_hits_last_5m > 0`     → `RateLimited`
    /// - `Disabled` is never changed here  (admin-only)
    pub fn record_error(&mut self, key_id: &str, is_rate_limit: bool) {
        let Some(key) = self.get_key_mut(key_id) else {
            tracing::warn!(key_id, "record_error: key not found");
            return;
        };

        // Never override an administratively disabled key.
        if key.health.status == KeyStatus::Disabled {
            return;
        }

        let now = Instant::now();
        if is_rate_limit {
            key.health.rate_limit_hits_last_5m += 1;
            key.health.last_error = Some((now, "rate_limited".into()));
            key.health.status = KeyStatus::RateLimited;
        } else {
            key.health.errors_last_5m += 1;
            key.health.last_error = Some((now, "error".into()));
            if key.health.errors_last_5m > DEGRADED_THRESHOLD {
                key.health.status = KeyStatus::Degraded;
            }
        }
    }

    /// Mark a key as rate-limited (convenience wrapper around `record_error`).
    pub fn record_rate_limit(&mut self, key_id: &str) {
        self.record_error(key_id, true);
    }

    /// Re-evaluate all keys' statuses based on current error counters.
    ///
    /// Call periodically (e.g., every minute) to allow keys to recover.
    ///
    /// Transition rules on recheck:
    ///
    /// - `Degraded` → `Healthy` if `errors_last_5m == 0`, stays `Degraded` if
    ///   still above threshold.
    /// - `RateLimited` → `Healthy` if `rate_limit_hits_last_5m == 0`; or
    ///   `Degraded` if `errors_last_5m > DEGRADED_THRESHOLD`.
    /// - `Disabled` is never changed automatically.
    /// - `Healthy` → `Degraded` if `errors_last_5m` drifted above threshold
    ///   (guards against races).
    pub fn recheck_health(&mut self) {
        for key in &mut self.keys {
            if key.health.status == KeyStatus::Disabled {
                continue;
            }

            let high_errors = key.health.errors_last_5m > DEGRADED_THRESHOLD;
            let rate_limited = key.health.rate_limit_hits_last_5m > 0;

            key.health.status = match (high_errors, rate_limited) {
                (_, true) => KeyStatus::RateLimited,
                (true, false) => KeyStatus::Degraded,
                (false, false) => KeyStatus::Healthy,
            };
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use http::{HeaderName, HeaderValue};
    use proptest::prelude::*;

    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::{KeyStatus, PooledKey};

    fn make_key(id: &str) -> PooledKey {
        PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer sk-test"),
                expires_at: None,
            },
            1.0,
        )
    }

    /// Stub selector that always returns the first eligible key.
    struct FirstEligible;
    impl KeySelector for FirstEligible {
        fn select<'a>(
            &self,
            pool: &'a [PooledKey],
            _req: &RequestContext,
        ) -> Option<&'a PooledKey> {
            pool.iter().find(|k| k.is_eligible())
        }
    }

    fn pool(keys: Vec<PooledKey>) -> KeyPool {
        KeyPool::new(keys, Box::new(FirstEligible))
    }

    /// Verifies `KeySelector` is object-safe.
    fn _assert_selector_object_safe(_: &dyn KeySelector) {}

    #[test]
    fn test_pool_len_and_empty() {
        let p = pool(vec![make_key("k1"), make_key("k2")]);
        assert_eq!(p.len(), 2);
        assert!(!p.is_empty());
        assert_eq!(pool(vec![]).len(), 0);
        assert!(pool(vec![]).is_empty());
    }

    #[test]
    fn test_eligible_count_excludes_disabled() {
        let mut keys = vec![make_key("k1"), make_key("k2"), make_key("k3")];
        keys[1].health.status = KeyStatus::Disabled;
        let p = pool(keys);
        assert_eq!(p.eligible_count(), 2);
    }

    #[test]
    fn test_select_returns_first_eligible() {
        let mut keys = vec![make_key("k1"), make_key("k2")];
        keys[0].health.status = KeyStatus::Disabled;
        let p = pool(keys);
        let ctx = RequestContext::default();
        let selected = p.select(&ctx).unwrap();
        assert_eq!(selected.id, "k2");
    }

    #[test]
    fn test_select_none_when_all_disabled() {
        let mut keys = vec![make_key("k1")];
        keys[0].health.status = KeyStatus::Disabled;
        let p = pool(keys);
        let ctx = RequestContext::default();
        assert!(p.select(&ctx).is_none());
    }

    #[test]
    fn test_select_none_on_empty_pool() {
        let p = pool(vec![]);
        let ctx = RequestContext::default();
        assert!(p.select(&ctx).is_none());
    }

    // ── add_key / remove_key tests ────────────────────────────────────────────

    #[test]
    fn test_add_key_increases_len() {
        let mut p = pool(vec![make_key("k1")]);
        p.add_key(make_key("k2"));
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn test_add_key_replaces_existing_id() {
        let mut p = pool(vec![make_key("k1")]);
        let mut replacement = make_key("k1");
        replacement.weight = 0.5;
        p.add_key(replacement);
        assert_eq!(p.len(), 1);
        assert!((p.keys[0].weight - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_remove_key_returns_true_on_success() {
        let mut p = pool(vec![make_key("k1"), make_key("k2")]);
        assert!(p.remove_key("k1"));
        assert_eq!(p.len(), 1);
        assert_eq!(p.keys[0].id, "k2");
    }

    #[test]
    fn test_remove_key_returns_false_when_not_found() {
        let mut p = pool(vec![make_key("k1")]);
        assert!(!p.remove_key("missing"));
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn test_get_key_mut_returns_correct_key() {
        let mut p = pool(vec![make_key("k1"), make_key("k2")]);
        let key = p.get_key_mut("k2").unwrap();
        assert_eq!(key.id, "k2");
    }

    #[test]
    fn test_get_key_mut_returns_none_for_missing() {
        let mut p = pool(vec![make_key("k1")]);
        assert!(p.get_key_mut("nope").is_none());
    }

    // ── record_success tests ──────────────────────────────────────────────────

    #[test]
    fn test_record_success_increments_requests() {
        let mut p = pool(vec![make_key("k1")]);
        p.record_success("k1", Duration::from_millis(100));
        assert_eq!(p.keys[0].health.total_requests, 1);
        assert!(p.keys[0].health.last_used.is_some());
    }

    #[test]
    fn test_record_success_updates_avg_latency_first_call() {
        let mut p = pool(vec![make_key("k1")]);
        p.record_success("k1", Duration::from_millis(200));
        assert!((p.keys[0].health.avg_latency_ms - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_record_success_ema_subsequent_calls() {
        let mut p = pool(vec![make_key("k1")]);
        p.record_success("k1", Duration::from_millis(100)); // sets avg = 100
        p.record_success("k1", Duration::from_millis(200)); // EMA update
        // EMA = 0.1 * 200 + 0.9 * 100 = 20 + 90 = 110
        assert!((p.keys[0].health.avg_latency_ms - 110.0).abs() < 1.0);
    }

    #[test]
    fn test_record_success_noop_on_missing_key() {
        let mut p = pool(vec![make_key("k1")]);
        // Should not panic.
        p.record_success("nope", Duration::from_millis(50));
    }

    // ── record_error tests ────────────────────────────────────────────────────

    #[test]
    fn test_record_error_increments_errors() {
        let mut p = pool(vec![make_key("k1")]);
        p.record_error("k1", false);
        assert_eq!(p.keys[0].health.errors_last_5m, 1);
        assert!(p.keys[0].health.last_error.is_some());
    }

    #[test]
    fn test_record_error_transitions_to_degraded_after_threshold() {
        let mut p = pool(vec![make_key("k1")]);
        for _ in 0..=DEGRADED_THRESHOLD {
            p.record_error("k1", false);
        }
        assert_eq!(p.keys[0].health.status, KeyStatus::Degraded);
    }

    #[test]
    fn test_record_error_rate_limit_transitions_to_rate_limited() {
        let mut p = pool(vec![make_key("k1")]);
        p.record_error("k1", true);
        assert_eq!(p.keys[0].health.status, KeyStatus::RateLimited);
        assert_eq!(p.keys[0].health.rate_limit_hits_last_5m, 1);
    }

    #[test]
    fn test_record_error_does_not_touch_disabled_key() {
        let mut p = pool(vec![make_key("k1")]);
        p.keys[0].health.status = KeyStatus::Disabled;
        p.record_error("k1", false);
        // Status must remain Disabled and error counter must not change.
        assert_eq!(p.keys[0].health.status, KeyStatus::Disabled);
        assert_eq!(p.keys[0].health.errors_last_5m, 0);
    }

    #[test]
    fn test_record_rate_limit_is_convenience_wrapper() {
        let mut p = pool(vec![make_key("k1")]);
        p.record_rate_limit("k1");
        assert_eq!(p.keys[0].health.status, KeyStatus::RateLimited);
    }

    // ── recheck_health tests ──────────────────────────────────────────────────

    #[test]
    fn test_recheck_health_recovers_degraded_when_errors_cleared() {
        let mut p = pool(vec![make_key("k1")]);
        p.keys[0].health.status = KeyStatus::Degraded;
        // errors cleared (simulated by a counter reset):
        p.keys[0].health.errors_last_5m = 0;
        p.recheck_health();
        assert_eq!(p.keys[0].health.status, KeyStatus::Healthy);
    }

    #[test]
    fn test_recheck_health_keeps_degraded_if_still_above_threshold() {
        let mut p = pool(vec![make_key("k1")]);
        p.keys[0].health.errors_last_5m = DEGRADED_THRESHOLD + 1;
        p.recheck_health();
        assert_eq!(p.keys[0].health.status, KeyStatus::Degraded);
    }

    #[test]
    fn test_recheck_health_recovers_rate_limited_when_hits_cleared() {
        let mut p = pool(vec![make_key("k1")]);
        p.keys[0].health.status = KeyStatus::RateLimited;
        p.keys[0].health.rate_limit_hits_last_5m = 0;
        p.recheck_health();
        assert_eq!(p.keys[0].health.status, KeyStatus::Healthy);
    }

    #[test]
    fn test_recheck_health_does_not_touch_disabled_key() {
        let mut p = pool(vec![make_key("k1")]);
        p.keys[0].health.status = KeyStatus::Disabled;
        p.recheck_health();
        assert_eq!(p.keys[0].health.status, KeyStatus::Disabled);
    }

    // ── Property tests ────────────────────────────────────────────────────────

    proptest! {
        /// A selector must never return a disabled key regardless of pool composition.
        #[test]
        fn prop_selector_never_returns_disabled(
            num_keys in 1usize..=10,
            disable_mask in proptest::collection::vec(any::<bool>(), 1..=10),
        ) {
            let keys: Vec<PooledKey> = (0..num_keys)
                .map(|i| {
                    let mut k = make_key(&format!("k{i}"));
                    let should_disable = disable_mask.get(i).copied().unwrap_or(false);
                    if should_disable {
                        k.health.status = KeyStatus::Disabled;
                    }
                    k
                })
                .collect();

            let p = pool(keys);
            let ctx = RequestContext::default();

            if let Some(selected) = p.select(&ctx) {
                prop_assert_ne!(&selected.health.status, &KeyStatus::Disabled);
            }
        }
    }

    /// WeightedRandom selection is thread-safe under concurrent access.
    #[test]
    fn prop_weighted_random_concurrent_access() {
        use crate::key_pool::selector::WeightedRandomSelector;
        use std::sync::Arc;

        let keys: Arc<Vec<PooledKey>> =
            Arc::new((0..5).map(|i| make_key(&format!("k{i}"))).collect());
        let sel = Arc::new(WeightedRandomSelector);
        let ctx = Arc::new(RequestContext::default());

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let sel = Arc::clone(&sel);
                let keys = Arc::clone(&keys);
                let ctx = Arc::clone(&ctx);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        let result = sel.select(&keys, &ctx);
                        assert!(result.is_some());
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }
}
