//! Weighted-random key selector.

use rand::Rng;
use switchboard_common::types::RequestContext;

use crate::key_pool::{KeySelector, PooledKey};

/// Selects a key by weighted random sampling over each key's `effective_weight()`.
///
/// Keys with higher configured weight (and healthy status) are proportionally
/// more likely to be selected.  Keys whose `effective_weight()` is `0.0`
/// (Disabled or RateLimited) are never chosen.
#[derive(Debug, Default)]
pub struct WeightedRandomSelector;

impl KeySelector for WeightedRandomSelector {
    fn select<'a>(
        &self,
        pool: &'a [PooledKey],
        _request: &RequestContext,
    ) -> Option<&'a PooledKey> {
        // Collect eligible keys with positive effective weight.
        let candidates: Vec<(usize, f64)> = pool
            .iter()
            .enumerate()
            .filter_map(|(i, k)| {
                let w = k.effective_weight();
                if w > 0.0 { Some((i, w)) } else { None }
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let total: f64 = candidates.iter().map(|(_, w)| w).sum();
        let mut rng = rand::thread_rng();
        let mut target = rng.r#gen::<f64>() * total;

        for (idx, weight) in &candidates {
            target -= weight;
            if target <= 0.0 {
                return Some(&pool[*idx]);
            }
        }

        // Fallback: floating-point edge case, return last candidate.
        candidates.last().map(|(idx, _)| &pool[*idx])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use http::{HeaderName, HeaderValue};

    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::{KeyStatus, PooledKey};

    fn make_key(id: &str, weight: f64) -> PooledKey {
        PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer sk-test"),
                expires_at: None,
            },
            weight,
        )
    }

    #[test]
    fn test_weighted_random_returns_none_on_empty() {
        let sel = WeightedRandomSelector;
        let ctx = RequestContext::default();
        assert!(sel.select(&[], &ctx).is_none());
    }

    #[test]
    fn test_weighted_random_skips_disabled_keys() {
        let mut k1 = make_key("k1", 1.0);
        k1.health.status = KeyStatus::Disabled;
        let k2 = make_key("k2", 1.0);
        let pool = vec![k1, k2];
        let sel = WeightedRandomSelector;
        let ctx = RequestContext::default();
        for _ in 0..20 {
            let chosen = sel.select(&pool, &ctx).unwrap();
            assert_eq!(chosen.id, "k2", "disabled key must never be selected");
        }
    }

    #[test]
    fn test_weighted_random_skips_rate_limited_keys() {
        let mut k1 = make_key("k1", 1.0);
        k1.health.status = KeyStatus::RateLimited;
        let k2 = make_key("k2", 1.0);
        let pool = vec![k1, k2];
        let sel = WeightedRandomSelector;
        let ctx = RequestContext::default();
        for _ in 0..20 {
            let chosen = sel.select(&pool, &ctx).unwrap();
            assert_eq!(chosen.id, "k2");
        }
    }

    #[test]
    fn test_weighted_random_only_eligible_key_always_selected() {
        let pool = vec![make_key("only", 0.5)];
        let sel = WeightedRandomSelector;
        let ctx = RequestContext::default();
        for _ in 0..10 {
            assert_eq!(sel.select(&pool, &ctx).unwrap().id, "only");
        }
    }

    #[test]
    fn test_weighted_random_returns_none_when_all_disabled() {
        let mut k1 = make_key("k1", 1.0);
        k1.health.status = KeyStatus::Disabled;
        let sel = WeightedRandomSelector;
        let ctx = RequestContext::default();
        assert!(sel.select(&[k1], &ctx).is_none());
    }

    #[test]
    fn test_weighted_random_concurrency() {
        use std::sync::Arc;

        let keys: Vec<PooledKey> = (0..5).map(|i| make_key(&format!("k{i}"), 1.0)).collect();
        let sel = Arc::new(WeightedRandomSelector);
        let ctx = Arc::new(RequestContext::default());
        let keys = Arc::new(keys);

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let sel = Arc::clone(&sel);
                let ctx = Arc::clone(&ctx);
                let keys = Arc::clone(&keys);
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
