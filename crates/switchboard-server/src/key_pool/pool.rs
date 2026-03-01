//! `KeySelector` trait and `KeyPool` full implementation.

use crate::key_pool::PooledKey;
use crate::key_pool::health::KeyStatus;
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
}
impl KeyPool {
    pub fn new(keys: Vec<Arc<RwLock<PooledKey>>>, selector: Box<dyn KeySelector>) -> Self {
        Self { keys, selector }
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
}
