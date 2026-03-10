//! Weighted-random key selector.

use crate::key_pool::{KeySelector, PooledKey};
use rand::Rng;
use std::sync::{Arc, RwLock};
use switchboard_common::types::RequestContext;

#[derive(Debug, Default)]
pub struct WeightedRandomSelector;

impl KeySelector for WeightedRandomSelector {
    fn select(
        &self,
        pool: &[Arc<RwLock<PooledKey>>],
        _r: &RequestContext,
    ) -> Option<Arc<RwLock<PooledKey>>> {
        let candidates: Vec<(usize, f64)> = pool
            .iter()
            .enumerate()
            .filter_map(|(i, k)| {
                let w = k.read().unwrap().effective_weight();
                if w > 0.0 { Some((i, w)) } else { None }
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let total: f64 = candidates.iter().map(|(_, w)| w).sum();
        let mut rng = rand::rng();
        let mut target = rng.random::<f64>() * total;
        for (idx, weight) in &candidates {
            target -= weight;
            if target <= 0.0 {
                return Some(Arc::clone(&pool[*idx]));
            }
        }
        candidates.last().map(|(idx, _)| Arc::clone(&pool[*idx]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::{KeyStatus, PooledKey};
    use http::{HeaderName, HeaderValue};
    fn mk(id: &str, w: f64) -> Arc<std::sync::RwLock<PooledKey>> {
        Arc::new(std::sync::RwLock::new(PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer sk-test"),
                expires_at: None,
            },
            w,
        )))
    }
    #[test]
    fn test_weighted_random_returns_none_on_empty() {
        assert!(
            WeightedRandomSelector
                .select(&[], &RequestContext::default())
                .is_none()
        );
    }
    #[test]
    fn test_weighted_random_skips_disabled_keys() {
        let k1 = mk("k1", 1.0);
        k1.write().unwrap().health.status = KeyStatus::Disabled;
        let k2 = mk("k2", 1.0);
        let pool = vec![k1, k2];
        for _ in 0..20 {
            assert_eq!(
                WeightedRandomSelector
                    .select(&pool, &RequestContext::default())
                    .unwrap()
                    .read()
                    .unwrap()
                    .id,
                "k2"
            );
        }
    }
    #[test]
    fn test_weighted_random_skips_rate_limited_keys() {
        let k1 = mk("k1", 1.0);
        k1.write().unwrap().health.status = KeyStatus::RateLimited;
        let k2 = mk("k2", 1.0);
        let pool = vec![k1, k2];
        for _ in 0..20 {
            assert_eq!(
                WeightedRandomSelector
                    .select(&pool, &RequestContext::default())
                    .unwrap()
                    .read()
                    .unwrap()
                    .id,
                "k2"
            );
        }
    }
    #[test]
    fn test_weighted_random_only_eligible_key_always_selected() {
        let pool = vec![mk("only", 0.5)];
        for _ in 0..10 {
            assert_eq!(
                WeightedRandomSelector
                    .select(&pool, &RequestContext::default())
                    .unwrap()
                    .read()
                    .unwrap()
                    .id,
                "only"
            );
        }
    }
    #[test]
    fn test_weighted_random_returns_none_when_all_disabled() {
        let k1 = mk("k1", 1.0);
        k1.write().unwrap().health.status = KeyStatus::Disabled;
        assert!(
            WeightedRandomSelector
                .select(&[k1], &RequestContext::default())
                .is_none()
        );
    }
    #[test]
    fn test_weighted_random_concurrency() {
        use std::sync::Arc;
        let keys: Vec<Arc<std::sync::RwLock<PooledKey>>> =
            (0..5).map(|i| mk(&format!("k{i}"), 1.0)).collect();
        let sel = Arc::new(WeightedRandomSelector);
        let ctx = Arc::new(RequestContext::default());
        let keys = Arc::new(keys);
        let hs: Vec<_> = (0..10)
            .map(|_| {
                let sel = Arc::clone(&sel);
                let ctx = Arc::clone(&ctx);
                let keys = Arc::clone(&keys);
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
