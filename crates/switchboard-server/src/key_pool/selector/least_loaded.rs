//! Least-loaded key selector.

use switchboard_common::types::RequestContext;

use crate::key_pool::{KeySelector, PooledKey};

/// Selects the eligible key with the lowest `health.total_requests`.
///
/// Ties are broken by pool order (first key with the minimum wins).
/// This strategy distributes load evenly over time when all keys start empty.
#[derive(Debug, Default)]
pub struct LeastLoadedSelector;

impl KeySelector for LeastLoadedSelector {
    fn select<'a>(
        &self,
        pool: &'a [PooledKey],
        _request: &RequestContext,
    ) -> Option<&'a PooledKey> {
        pool.iter()
            .filter(|k| k.is_eligible())
            .min_by_key(|k| k.health.total_requests)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use http::{HeaderName, HeaderValue};

    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::{KeyStatus, PooledKey};

    fn make_key(id: &str, total_requests: u64) -> PooledKey {
        let mut k = PooledKey::new_static(
            id,
            UpstreamCredentials {
                header_name: HeaderName::from_static("authorization"),
                header_value: HeaderValue::from_static("Bearer sk-test"),
                expires_at: None,
            },
            1.0,
        );
        k.health.total_requests = total_requests;
        k
    }

    #[test]
    fn test_least_loaded_selects_minimum() {
        let pool = vec![make_key("k1", 100), make_key("k2", 5), make_key("k3", 50)];
        let sel = LeastLoadedSelector;
        let ctx = RequestContext::default();
        assert_eq!(sel.select(&pool, &ctx).unwrap().id, "k2");
    }

    #[test]
    fn test_least_loaded_breaks_ties_by_order() {
        let pool = vec![make_key("k1", 10), make_key("k2", 10), make_key("k3", 20)];
        let sel = LeastLoadedSelector;
        let ctx = RequestContext::default();
        // k1 and k2 tie; k1 appears first.
        assert_eq!(sel.select(&pool, &ctx).unwrap().id, "k1");
    }

    #[test]
    fn test_least_loaded_skips_disabled() {
        let mut k1 = make_key("k1", 1);
        k1.health.status = KeyStatus::Disabled;
        let k2 = make_key("k2", 100);
        let pool = vec![k1, k2];
        let sel = LeastLoadedSelector;
        let ctx = RequestContext::default();
        // k1 is disabled, so k2 must be returned despite higher load.
        assert_eq!(sel.select(&pool, &ctx).unwrap().id, "k2");
    }

    #[test]
    fn test_least_loaded_empty_returns_none() {
        let sel = LeastLoadedSelector;
        let ctx = RequestContext::default();
        assert!(sel.select(&[], &ctx).is_none());
    }

    #[test]
    fn test_least_loaded_all_disabled_returns_none() {
        let mut k = make_key("k1", 0);
        k.health.status = KeyStatus::Disabled;
        let sel = LeastLoadedSelector;
        let ctx = RequestContext::default();
        assert!(sel.select(&[k], &ctx).is_none());
    }
}
