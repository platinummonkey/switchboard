//! `KeySelector` trait and `KeyPool` full implementation.

use crate::key_pool::PooledKey;
use crate::key_pool::health::KeyStatus;
use dashmap::DashMap;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use switchboard_common::types::RequestContext;

pub trait KeySelector: Send + Sync {
    fn select(
        &self,
        pool: &[Arc<RwLock<PooledKey>>],
        request: &RequestContext,
    ) -> Option<Arc<RwLock<PooledKey>>>;
}

const DEGRADED_THRESHOLD: u64 = 5;

pub struct KeyPool {
    pub(crate) keys: Vec<Arc<RwLock<PooledKey>>>,
    pub(crate) selector: Box<dyn KeySelector>,
    in_flight: Option<Arc<DashMap<String, AtomicU32>>>,
}
impl KeyPool {
    pub fn new(keys: Vec<Arc<RwLock<PooledKey>>>, selector: Box<dyn KeySelector>) -> Self {
        Self {
            keys,
            selector,
            in_flight: None,
        }
    }
    /// Construct a pool that shares an in-flight tracker with an external
    /// component (e.g. [`LeastLoadedSelector`]).  The tracker is used by
    /// [`AuthInjectService`] to increment/decrement in-flight counters around
    /// each proxied request so the selector always sees live load figures.
    pub fn new_with_tracker(
        keys: Vec<Arc<RwLock<PooledKey>>>,
        selector: Box<dyn KeySelector>,
        tracker: Option<Arc<DashMap<String, AtomicU32>>>,
    ) -> Self {
        Self {
            keys,
            selector,
            in_flight: tracker,
        }
    }
    /// Return a clone of the shared in-flight tracker, if one was registered.
    ///
    /// Returns `Some` only for pools that use [`LeastLoadedSelector`] (or any
    /// other selector that was wired with a tracker at construction time).
    /// Returns `None` for all other pools — callers treat that as "no-op".
    pub fn in_flight_tracker(&self) -> Option<Arc<DashMap<String, AtomicU32>>> {
        self.in_flight.clone()
    }
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    pub fn eligible_count(&self) -> usize {
        self.keys
            .iter()
            .filter(|k| k.read().unwrap().is_eligible())
            .count()
    }
    pub fn select(&self, request: &RequestContext) -> Option<Arc<RwLock<PooledKey>>> {
        self.selector.select(&self.keys, request)
    }
    pub fn add_key(&mut self, key: PooledKey) {
        let id = key.id.clone();
        if let Some(e) = self.keys.iter().find(|k| k.read().unwrap().id == id) {
            *e.write().unwrap() = key;
        } else {
            self.keys.push(Arc::new(RwLock::new(key)));
        }
    }
    pub fn remove_key(&mut self, id: &str) -> bool {
        if let Some(pos) = self.keys.iter().position(|k| k.read().unwrap().id == id) {
            self.keys.remove(pos);
            true
        } else {
            false
        }
    }
    pub fn get_key(&self, id: &str) -> Option<Arc<RwLock<PooledKey>>> {
        self.keys
            .iter()
            .find(|k| k.read().unwrap().id == id)
            .map(Arc::clone)
    }
    pub fn get_key_mut(&self, id: &str) -> Option<Arc<RwLock<PooledKey>>> {
        self.get_key(id)
    }
    pub fn record_success(&self, key_id: &str, latency: Duration) {
        let Some(arc) = self.get_key(key_id) else {
            tracing::warn!(key_id, "record_success: key not found");
            return;
        };
        let mut key = arc.write().unwrap();
        key.health.total_requests += 1;
        key.health.last_used = Some(Instant::now());
        let ms = latency.as_secs_f64() * 1000.0;
        if key.health.total_requests == 1 {
            key.health.avg_latency_ms = ms;
        } else {
            key.health.avg_latency_ms = 0.1 * ms + 0.9 * key.health.avg_latency_ms;
        }
    }
    pub fn record_error(&self, key_id: &str, is_rate_limit: bool) {
        let Some(arc) = self.get_key(key_id) else {
            tracing::warn!(key_id, "record_error: key not found");
            return;
        };
        let mut key = arc.write().unwrap();
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
    pub fn record_rate_limit(&self, key_id: &str) {
        self.record_error(key_id, true);
    }
    pub fn recheck_health(&self) {
        for arc in &self.keys {
            let mut key = arc.write().unwrap();
            if key.health.status == KeyStatus::Disabled {
                continue;
            }
            let high = key.health.errors_last_5m > DEGRADED_THRESHOLD;
            let rl = key.health.rate_limit_hits_last_5m > 0;
            key.health.status = match (high, rl) {
                (_, true) => KeyStatus::RateLimited,
                (true, false) => KeyStatus::Degraded,
                _ => KeyStatus::Healthy,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::{KeyStatus, PooledKey};
    use http::{HeaderName, HeaderValue};
    use proptest::prelude::*;
    fn mk(id: &str) -> Arc<RwLock<PooledKey>> {
        Arc::new(RwLock::new(PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer sk-test"),
                expires_at: None,
            },
            1.0,
        )))
    }
    fn mkr(id: &str) -> PooledKey {
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
    struct FE;
    impl KeySelector for FE {
        fn select(
            &self,
            pool: &[Arc<RwLock<PooledKey>>],
            _r: &RequestContext,
        ) -> Option<Arc<RwLock<PooledKey>>> {
            pool.iter()
                .find(|k| k.read().unwrap().is_eligible())
                .map(Arc::clone)
        }
    }
    fn pool(keys: Vec<Arc<RwLock<PooledKey>>>) -> KeyPool {
        KeyPool::new(keys, Box::new(FE))
    }
    fn _safe(_: &dyn KeySelector) {}
    #[test]
    fn test_pool_len_and_empty() {
        let p = pool(vec![mk("k1"), mk("k2")]);
        assert_eq!(p.len(), 2);
        assert!(!p.is_empty());
        assert_eq!(pool(vec![]).len(), 0);
        assert!(pool(vec![]).is_empty());
    }
    #[test]
    fn test_eligible_count_excludes_disabled() {
        let keys = vec![mk("k1"), mk("k2"), mk("k3")];
        keys[1].write().unwrap().health.status = KeyStatus::Disabled;
        assert_eq!(pool(keys).eligible_count(), 2);
    }
    #[test]
    fn test_select_returns_first_eligible() {
        let keys = vec![mk("k1"), mk("k2")];
        keys[0].write().unwrap().health.status = KeyStatus::Disabled;
        assert_eq!(
            pool(keys)
                .select(&RequestContext::default())
                .unwrap()
                .read()
                .unwrap()
                .id,
            "k2"
        );
    }
    #[test]
    fn test_select_none_when_all_disabled() {
        let keys = vec![mk("k1")];
        keys[0].write().unwrap().health.status = KeyStatus::Disabled;
        assert!(pool(keys).select(&RequestContext::default()).is_none());
    }
    #[test]
    fn test_select_none_on_empty_pool() {
        assert!(pool(vec![]).select(&RequestContext::default()).is_none());
    }
    #[test]
    fn test_add_key_increases_len() {
        let mut p = pool(vec![mk("k1")]);
        p.add_key(mkr("k2"));
        assert_eq!(p.len(), 2);
    }
    #[test]
    fn test_add_key_replaces_existing_id() {
        let mut p = pool(vec![mk("k1")]);
        let mut r = mkr("k1");
        r.weight = 0.5;
        p.add_key(r);
        assert_eq!(p.len(), 1);
        assert!((p.keys[0].read().unwrap().weight - 0.5).abs() < f64::EPSILON);
    }
    #[test]
    fn test_remove_key_returns_true_on_success() {
        let mut p = pool(vec![mk("k1"), mk("k2")]);
        assert!(p.remove_key("k1"));
        assert_eq!(p.keys[0].read().unwrap().id, "k2");
    }
    #[test]
    fn test_remove_key_returns_false_when_not_found() {
        assert!(!pool(vec![mk("k1")]).remove_key("missing"));
    }
    #[test]
    fn test_get_key_mut_returns_correct_key() {
        let p = pool(vec![mk("k1"), mk("k2")]);
        assert_eq!(p.get_key_mut("k2").unwrap().read().unwrap().id, "k2");
    }
    #[test]
    fn test_get_key_mut_returns_none_for_missing() {
        assert!(pool(vec![mk("k1")]).get_key_mut("nope").is_none());
    }
    #[test]
    fn test_record_success_increments_requests() {
        let p = pool(vec![mk("k1")]);
        p.record_success("k1", Duration::from_millis(100));
        assert_eq!(p.keys[0].read().unwrap().health.total_requests, 1);
        assert!(p.keys[0].read().unwrap().health.last_used.is_some());
    }
    #[test]
    fn test_record_success_updates_avg_latency_first_call() {
        let p = pool(vec![mk("k1")]);
        p.record_success("k1", Duration::from_millis(200));
        assert!((p.keys[0].read().unwrap().health.avg_latency_ms - 200.0).abs() < 1.0);
    }
    #[test]
    fn test_record_success_ema_subsequent_calls() {
        let p = pool(vec![mk("k1")]);
        p.record_success("k1", Duration::from_millis(100));
        p.record_success("k1", Duration::from_millis(200));
        assert!((p.keys[0].read().unwrap().health.avg_latency_ms - 110.0).abs() < 1.0);
    }
    #[test]
    fn test_record_success_noop_on_missing_key() {
        pool(vec![mk("k1")]).record_success("nope", Duration::from_millis(50));
    }
    #[test]
    fn test_record_error_increments_errors() {
        let p = pool(vec![mk("k1")]);
        p.record_error("k1", false);
        assert_eq!(p.keys[0].read().unwrap().health.errors_last_5m, 1);
        assert!(p.keys[0].read().unwrap().health.last_error.is_some());
    }
    #[test]
    fn test_record_error_transitions_to_degraded_after_threshold() {
        let p = pool(vec![mk("k1")]);
        for _ in 0..=DEGRADED_THRESHOLD {
            p.record_error("k1", false);
        }
        assert_eq!(p.keys[0].read().unwrap().health.status, KeyStatus::Degraded);
    }
    #[test]
    fn test_record_error_rate_limit_transitions_to_rate_limited() {
        let p = pool(vec![mk("k1")]);
        p.record_error("k1", true);
        assert_eq!(
            p.keys[0].read().unwrap().health.status,
            KeyStatus::RateLimited
        );
        assert_eq!(p.keys[0].read().unwrap().health.rate_limit_hits_last_5m, 1);
    }
    #[test]
    fn test_record_error_does_not_touch_disabled_key() {
        let p = pool(vec![mk("k1")]);
        p.keys[0].write().unwrap().health.status = KeyStatus::Disabled;
        p.record_error("k1", false);
        assert_eq!(p.keys[0].read().unwrap().health.status, KeyStatus::Disabled);
        assert_eq!(p.keys[0].read().unwrap().health.errors_last_5m, 0);
    }
    #[test]
    fn test_record_rate_limit_is_convenience_wrapper() {
        let p = pool(vec![mk("k1")]);
        p.record_rate_limit("k1");
        assert_eq!(
            p.keys[0].read().unwrap().health.status,
            KeyStatus::RateLimited
        );
    }
    #[test]
    fn test_recheck_health_recovers_degraded_when_errors_cleared() {
        let p = pool(vec![mk("k1")]);
        p.keys[0].write().unwrap().health.status = KeyStatus::Degraded;
        p.keys[0].write().unwrap().health.errors_last_5m = 0;
        p.recheck_health();
        assert_eq!(p.keys[0].read().unwrap().health.status, KeyStatus::Healthy);
    }
    #[test]
    fn test_recheck_health_keeps_degraded_if_still_above_threshold() {
        let p = pool(vec![mk("k1")]);
        p.keys[0].write().unwrap().health.errors_last_5m = DEGRADED_THRESHOLD + 1;
        p.recheck_health();
        assert_eq!(p.keys[0].read().unwrap().health.status, KeyStatus::Degraded);
    }
    #[test]
    fn test_recheck_health_recovers_rate_limited_when_hits_cleared() {
        let p = pool(vec![mk("k1")]);
        p.keys[0].write().unwrap().health.status = KeyStatus::RateLimited;
        p.keys[0].write().unwrap().health.rate_limit_hits_last_5m = 0;
        p.recheck_health();
        assert_eq!(p.keys[0].read().unwrap().health.status, KeyStatus::Healthy);
    }
    #[test]
    fn test_recheck_health_does_not_touch_disabled_key() {
        let p = pool(vec![mk("k1")]);
        p.keys[0].write().unwrap().health.status = KeyStatus::Disabled;
        p.recheck_health();
        assert_eq!(p.keys[0].read().unwrap().health.status, KeyStatus::Disabled);
    }
    proptest! {
        #[test] fn prop_selector_never_returns_disabled(num_keys in 1usize..=10, disable_mask in proptest::collection::vec(any::<bool>(), 1..=10)) {
            let keys: Vec<Arc<RwLock<PooledKey>>> = (0..num_keys).map(|i| { let k = mk(&format!("k{i}")); if disable_mask.get(i).copied().unwrap_or(false) { k.write().unwrap().health.status = KeyStatus::Disabled; } k }).collect();
            let p = pool(keys);
            if let Some(s) = p.select(&RequestContext::default()) { prop_assert_ne!(&s.read().unwrap().health.status, &KeyStatus::Disabled); }
        }
    }
    #[test]
    fn prop_weighted_random_concurrent_access() {
        use crate::key_pool::selector::WeightedRandomSelector;
        let keys: Arc<Vec<Arc<RwLock<PooledKey>>>> =
            Arc::new((0..5).map(|i| mk(&format!("k{i}"))).collect());
        let sel = Arc::new(WeightedRandomSelector);
        let ctx = Arc::new(RequestContext::default());
        let hs: Vec<_> = (0..10)
            .map(|_| {
                let sel = Arc::clone(&sel);
                let keys = Arc::clone(&keys);
                let ctx = Arc::clone(&ctx);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        assert!(sel.select(&keys, &ctx).is_some());
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    }

    // ── in_flight_tracker wiring tests ──────────────────────────────────────

    #[test]
    fn test_in_flight_tracker_none_for_plain_pool() {
        let p = pool(vec![mk("k1")]);
        assert!(p.in_flight_tracker().is_none());
    }

    #[test]
    fn test_in_flight_tracker_some_when_tracker_provided() {
        use crate::key_pool::selector::LeastLoadedSelector;

        let sel = LeastLoadedSelector::new();
        let tracker = sel.in_flight_tracker();
        let p = KeyPool::new_with_tracker(vec![mk("k1"), mk("k2")], Box::new(sel), Some(tracker));
        assert!(p.in_flight_tracker().is_some());
    }

    /// Simulate what `AuthInjectService` does: after `select()` returns a key,
    /// increment the shared tracker.  Verify the counter is 1 and that a
    /// subsequent `select()` on the same pool now prefers the other key.
    #[test]
    fn test_in_flight_counter_incremented_via_tracker_shifts_selection() {
        use crate::key_pool::selector::LeastLoadedSelector;
        use std::sync::atomic::Ordering;

        let sel = LeastLoadedSelector::new();
        let tracker = sel.in_flight_tracker();
        let p = KeyPool::new_with_tracker(
            vec![mk("k1"), mk("k2")],
            Box::new(sel),
            Some(Arc::clone(&tracker)),
        );

        // First selection — no load recorded yet, should return k1 (first).
        let first = p.select(&RequestContext::default()).unwrap();
        let key_id = first.read().unwrap().id.clone();
        assert_eq!(key_id, "k1");

        // Simulate AuthInjectService incrementing the counter for the selected key.
        let pool_tracker = p.in_flight_tracker().unwrap();
        pool_tracker
            .entry(key_id.clone())
            .or_insert_with(|| AtomicU32::new(0))
            .fetch_add(1, Ordering::Relaxed);

        // Counter for k1 should now be 1.
        assert_eq!(pool_tracker.get("k1").unwrap().load(Ordering::Relaxed), 1);

        // Second selection — k1 has 1 in-flight, k2 has 0, so k2 is selected.
        let second = p.select(&RequestContext::default()).unwrap();
        assert_eq!(second.read().unwrap().id, "k2");

        // Simulate decrement (request complete).
        if let Some(entry) = pool_tracker.get("k1") {
            entry.fetch_sub(1, Ordering::Relaxed);
        }
        assert_eq!(pool_tracker.get("k1").unwrap().load(Ordering::Relaxed), 0);
    }
}
