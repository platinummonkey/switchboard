//! Round-robin key selector.

use crate::key_pool::{KeySelector, PooledKey};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use switchboard_common::types::RequestContext;

#[derive(Debug, Default)]
pub struct RoundRobinSelector {
    cursor: AtomicUsize,
}
impl RoundRobinSelector {
    pub fn new() -> Self {
        Self {
            cursor: AtomicUsize::new(0),
        }
    }
}

impl KeySelector for RoundRobinSelector {
    fn select(
        &self,
        pool: &[Arc<RwLock<PooledKey>>],
        _r: &RequestContext,
    ) -> Option<Arc<RwLock<PooledKey>>> {
        let eligible: Vec<usize> = pool
            .iter()
            .enumerate()
            .filter(|(_, k)| k.read().unwrap().is_eligible())
            .map(|(i, _)| i)
            .collect();
        if eligible.is_empty() {
            return None;
        }
        Some(Arc::clone(
            &pool[eligible[self.cursor.fetch_add(1, Ordering::Relaxed) % eligible.len()]],
        ))
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
    #[test]
    fn test_round_robin_cycles_through_keys() {
        let pool = vec![mk("k1"), mk("k2"), mk("k3")];
        let sel = RoundRobinSelector::new();
        let ctx = RequestContext::default();
        let ids: Vec<String> = (0..6)
            .map(|_| sel.select(&pool, &ctx).unwrap().read().unwrap().id.clone())
            .collect();
        assert_eq!(ids, vec!["k1", "k2", "k3", "k1", "k2", "k3"]);
    }
    #[test]
    fn test_round_robin_skips_disabled() {
        let keys = vec![mk("k1"), mk("k2"), mk("k3")];
        keys[1].write().unwrap().health.status = KeyStatus::Disabled;
        let sel = RoundRobinSelector::new();
        let ctx = RequestContext::default();
        let ids: Vec<String> = (0..4)
            .map(|_| sel.select(&keys, &ctx).unwrap().read().unwrap().id.clone())
            .collect();
        assert_eq!(ids, vec!["k1", "k3", "k1", "k3"]);
    }
    #[test]
    fn test_round_robin_empty_pool_returns_none() {
        assert!(
            RoundRobinSelector::new()
                .select(&[], &RequestContext::default())
                .is_none()
        );
    }
    #[test]
    fn test_round_robin_all_disabled_returns_none() {
        let k1 = mk("k1");
        k1.write().unwrap().health.status = KeyStatus::Disabled;
        assert!(
            RoundRobinSelector::new()
                .select(&[k1], &RequestContext::default())
                .is_none()
        );
    }
    #[test]
    fn test_round_robin_thread_safe() {
        use std::sync::Arc;
        let keys: Vec<Arc<RwLock<PooledKey>>> = (0..3).map(|i| mk(&format!("k{i}"))).collect();
        let sel = Arc::new(RoundRobinSelector::new());
        let keys = Arc::new(keys);
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
