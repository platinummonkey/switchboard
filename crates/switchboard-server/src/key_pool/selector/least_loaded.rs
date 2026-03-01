//! Least-loaded key selector with in-flight tracking.

use crate::key_pool::{KeySelector, PooledKey};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use switchboard_common::types::RequestContext;

#[derive(Debug, Default)]
pub struct LeastLoadedSelector {
    in_flight: Arc<DashMap<String, AtomicU32>>,
}
impl LeastLoadedSelector {
    pub fn new() -> Self {
        Self {
            in_flight: Arc::new(DashMap::new()),
        }
    }
    pub fn in_flight_tracker(&self) -> Arc<DashMap<String, AtomicU32>> {
        Arc::clone(&self.in_flight)
    }
}
impl KeySelector for LeastLoadedSelector {
    fn select(
        &self,
        pool: &[Arc<RwLock<PooledKey>>],
        _r: &RequestContext,
    ) -> Option<Arc<RwLock<PooledKey>>> {
        pool.iter()
            .filter(|k| k.read().unwrap().is_eligible())
            .min_by_key(|k| {
                let id = k.read().unwrap().id.clone();
                self.in_flight
                    .get(&id)
                    .map(|v| v.load(Ordering::Relaxed))
                    .unwrap_or(0)
            })
            .map(Arc::clone)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::{KeyStatus, PooledKey};
    use http::{HeaderName, HeaderValue};
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
    fn ms(pairs: &[(&str, u32)]) -> LeastLoadedSelector {
        let s = LeastLoadedSelector::new();
        for (id, c) in pairs {
            s.in_flight.insert(id.to_string(), AtomicU32::new(*c));
        }
        s
    }
    #[test]
    fn test_least_loaded_selects_minimum() {
        let pool = vec![mk("k1"), mk("k2"), mk("k3")];
        assert_eq!(
            ms(&[("k1", 100), ("k2", 5), ("k3", 50)])
                .select(&pool, &RequestContext::default())
                .unwrap()
                .read()
                .unwrap()
                .id,
            "k2"
        );
    }
    #[test]
    fn test_least_loaded_breaks_ties_by_order() {
        let pool = vec![mk("k1"), mk("k2"), mk("k3")];
        assert_eq!(
            ms(&[("k1", 10), ("k2", 10), ("k3", 20)])
                .select(&pool, &RequestContext::default())
                .unwrap()
                .read()
                .unwrap()
                .id,
            "k1"
        );
    }
    #[test]
    fn test_least_loaded_skips_disabled() {
        let pool = vec![mk("k1"), mk("k2")];
        pool[0].write().unwrap().health.status = KeyStatus::Disabled;
        assert_eq!(
            ms(&[("k1", 1), ("k2", 100)])
                .select(&pool, &RequestContext::default())
                .unwrap()
                .read()
                .unwrap()
                .id,
            "k2"
        );
    }
    #[test]
    fn test_least_loaded_empty_returns_none() {
        assert!(
            LeastLoadedSelector::new()
                .select(&[], &RequestContext::default())
                .is_none()
        );
    }
    #[test]
    fn test_least_loaded_all_disabled_returns_none() {
        let k = mk("k1");
        k.write().unwrap().health.status = KeyStatus::Disabled;
        assert!(
            LeastLoadedSelector::new()
                .select(&[k], &RequestContext::default())
                .is_none()
        );
    }
    #[test]
    fn test_least_loaded_no_in_flight_data_treats_as_zero() {
        let pool = vec![mk("k1"), mk("k2")];
        assert_eq!(
            LeastLoadedSelector::new()
                .select(&pool, &RequestContext::default())
                .unwrap()
                .read()
                .unwrap()
                .id,
            "k1"
        );
    }
    #[test]
    fn test_in_flight_tracker_increments_affect_selection() {
        let pool = vec![mk("k1"), mk("k2")];
        let s = LeastLoadedSelector::new();
        s.in_flight_tracker()
            .entry("k1".to_string())
            .or_insert_with(|| AtomicU32::new(0))
            .fetch_add(5, Ordering::Relaxed);
        assert_eq!(
            s.select(&pool, &RequestContext::default())
                .unwrap()
                .read()
                .unwrap()
                .id,
            "k2"
        );
    }
}
