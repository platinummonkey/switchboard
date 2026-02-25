//! Sticky key selector — consistently routes the same user to the same key.

use dashmap::DashMap;
use switchboard_common::types::RequestContext;

use crate::key_pool::{KeySelector, PooledKey};

/// Wraps an inner selector and provides per-user key stickiness.
///
/// On the first request for a `user_id`, the inner selector is consulted and
/// its result is cached in a `DashMap<user_id, key_id>`.  Subsequent requests
/// from the same user always return the same key, as long as it remains
/// eligible.  If the cached key is no longer eligible (disabled, removed), the
/// inner selector is called again and the mapping is refreshed.
///
/// Falls back to the inner selector when `request.user_id` is `None`.
pub struct StickySelector {
    inner: Box<dyn KeySelector>,
    /// user_id → key_id affinity map.
    map: DashMap<String, String>,
}

impl std::fmt::Debug for StickySelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StickySelector")
            .field("entries", &self.map.len())
            .finish()
    }
}

impl StickySelector {
    pub fn new(inner: Box<dyn KeySelector>) -> Self {
        Self {
            inner,
            map: DashMap::new(),
        }
    }
}

impl KeySelector for StickySelector {
    fn select<'a>(&self, pool: &'a [PooledKey], request: &RequestContext) -> Option<&'a PooledKey> {
        let Some(user_id) = request.user_id.as_deref() else {
            // No user identity: fall back to the inner selector without caching.
            return self.inner.select(pool, request);
        };

        // Check if we have a cached mapping and the key is still eligible.
        if let Some(key_id) = self.map.get(user_id) {
            if let Some(key) = pool.iter().find(|k| k.id == *key_id && k.is_eligible()) {
                return Some(key);
            }
            // Cached key is gone or ineligible: drop the stale mapping.
            drop(key_id);
            self.map.remove(user_id);
        }

        // No valid cached key: delegate to inner selector and cache the result.
        let key = self.inner.select(pool, request)?;
        self.map.insert(user_id.to_string(), key.id.clone());
        Some(key)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use http::{HeaderName, HeaderValue};

    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::selector::RoundRobinSelector;
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

    fn ctx_with_user(user_id: &str) -> RequestContext {
        RequestContext {
            user_id: Some(user_id.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn test_sticky_same_user_gets_same_key() {
        let pool = vec![make_key("k1"), make_key("k2"), make_key("k3")];
        let inner = Box::new(RoundRobinSelector::new());
        let sel = StickySelector::new(inner);
        let ctx = ctx_with_user("alice");

        let first = sel.select(&pool, &ctx).unwrap().id.clone();
        for _ in 0..9 {
            let id = sel.select(&pool, &ctx).unwrap().id.clone();
            assert_eq!(
                id, first,
                "sticky: same user should always get the same key"
            );
        }
    }

    #[test]
    fn test_sticky_different_users_may_get_different_keys() {
        let pool = vec![make_key("k1"), make_key("k2"), make_key("k3")];
        let inner = Box::new(RoundRobinSelector::new());
        let sel = StickySelector::new(inner);

        let id_alice = sel
            .select(&pool, &ctx_with_user("alice"))
            .unwrap()
            .id
            .clone();
        let id_bob = sel.select(&pool, &ctx_with_user("bob")).unwrap().id.clone();
        let id_carol = sel
            .select(&pool, &ctx_with_user("carol"))
            .unwrap()
            .id
            .clone();

        // With round-robin and 3 keys, the three users get k1, k2, k3 respectively.
        assert_eq!(id_alice, "k1");
        assert_eq!(id_bob, "k2");
        assert_eq!(id_carol, "k3");
    }

    #[test]
    fn test_sticky_fallback_on_stale_key() {
        let mut pool = vec![make_key("k1"), make_key("k2")];
        let inner = Box::new(RoundRobinSelector::new());
        let sel = StickySelector::new(inner);
        let ctx = ctx_with_user("alice");

        // Alice is assigned k1.
        let first = sel.select(&pool, &ctx).unwrap().id.clone();
        assert_eq!(first, "k1");

        // Disable k1: the cached mapping is now stale.
        pool[0].health.status = KeyStatus::Disabled;

        // Next call should fall back to inner selector and return k2.
        let fallback = sel.select(&pool, &ctx).unwrap().id.clone();
        assert_eq!(fallback, "k2");
    }

    #[test]
    fn test_sticky_no_user_falls_back_to_inner() {
        let pool = vec![make_key("k1"), make_key("k2")];
        let inner = Box::new(RoundRobinSelector::new());
        let sel = StickySelector::new(inner);
        let ctx = RequestContext::default(); // no user_id

        // Should still return a key via the inner selector.
        assert!(sel.select(&pool, &ctx).is_some());
    }

    #[test]
    fn test_sticky_empty_pool_returns_none() {
        let inner = Box::new(RoundRobinSelector::new());
        let sel = StickySelector::new(inner);
        assert!(sel.select(&[], &ctx_with_user("alice")).is_none());
    }
}
