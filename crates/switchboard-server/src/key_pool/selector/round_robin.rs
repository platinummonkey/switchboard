//! Round-robin key selector.

use std::sync::atomic::{AtomicUsize, Ordering};

use switchboard_common::types::RequestContext;

use crate::key_pool::{KeySelector, PooledKey};

/// Selects eligible keys in a round-robin order.
///
/// Uses an `AtomicUsize` cursor that is incremented on each call, so selection
/// is thread-safe without any locking.  Disabled keys are skipped.
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
    fn select<'a>(
        &self,
        pool: &'a [PooledKey],
        _request: &RequestContext,
    ) -> Option<&'a PooledKey> {
        // Collect eligible key indices.
        let eligible: Vec<usize> = pool
            .iter()
            .enumerate()
            .filter(|(_, k)| k.is_eligible())
            .map(|(i, _)| i)
            .collect();

        if eligible.is_empty() {
            return None;
        }

        // Atomically advance the cursor and wrap around the eligible count.
        let pos = self.cursor.fetch_add(1, Ordering::Relaxed) % eligible.len();
        Some(&pool[eligible[pos]])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use http::{HeaderName, HeaderValue};

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

    #[test]
    fn test_round_robin_cycles_through_keys() {
        let pool = vec![make_key("k1"), make_key("k2"), make_key("k3")];
        let sel = RoundRobinSelector::new();
        let ctx = RequestContext::default();

        let ids: Vec<&str> = (0..6)
            .map(|_| sel.select(&pool, &ctx).unwrap().id.as_str())
            .collect();

        assert_eq!(ids, vec!["k1", "k2", "k3", "k1", "k2", "k3"]);
    }

    #[test]
    fn test_round_robin_skips_disabled() {
        let mut keys = vec![make_key("k1"), make_key("k2"), make_key("k3")];
        keys[1].health.status = KeyStatus::Disabled;
        let sel = RoundRobinSelector::new();
        let ctx = RequestContext::default();

        let ids: Vec<&str> = (0..4)
            .map(|_| sel.select(&keys, &ctx).unwrap().id.as_str())
            .collect();

        // Only k1 and k3 are eligible.
        assert_eq!(ids, vec!["k1", "k3", "k1", "k3"]);
    }

    #[test]
    fn test_round_robin_empty_pool_returns_none() {
        let sel = RoundRobinSelector::new();
        let ctx = RequestContext::default();
        assert!(sel.select(&[], &ctx).is_none());
    }

    #[test]
    fn test_round_robin_all_disabled_returns_none() {
        let mut k1 = make_key("k1");
        k1.health.status = KeyStatus::Disabled;
        let sel = RoundRobinSelector::new();
        let ctx = RequestContext::default();
        assert!(sel.select(&[k1], &ctx).is_none());
    }

    #[test]
    fn test_round_robin_thread_safe() {
        use std::sync::Arc;

        let keys: Vec<PooledKey> = (0..3).map(|i| make_key(&format!("k{i}"))).collect();
        let sel = Arc::new(RoundRobinSelector::new());
        let keys = Arc::new(keys);
        let ctx = Arc::new(RequestContext::default());

        let handles: Vec<_> = (0..10)
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

        for h in handles {
            h.join().unwrap();
        }
    }
}
