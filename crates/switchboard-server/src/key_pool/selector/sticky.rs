//! Sticky key selector.

use crate::key_pool::{KeySelector, PooledKey};
use dashmap::DashMap;
use std::sync::{Arc, RwLock};
use switchboard_common::types::RequestContext;

pub struct StickySelector {
    inner: Box<dyn KeySelector>,
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
    fn select(
        &self,
        pool: &[Arc<RwLock<PooledKey>>],
        request: &RequestContext,
    ) -> Option<Arc<RwLock<PooledKey>>> {
        let Some(uid) = request.user_id.as_deref() else {
            return self.inner.select(pool, request);
        };
        if let Some(kid) = self.map.get(uid) {
            if let Some(arc) = pool
                .iter()
                .find(|k| {
                    let k = k.read().unwrap();
                    k.id == *kid && k.is_eligible()
                })
                .map(Arc::clone)
            {
                return Some(arc);
            }
            drop(kid);
            self.map.remove(uid);
        }
        let ka = self.inner.select(pool, request)?;
        let kid = ka.read().unwrap().id.clone();
        self.map.insert(uid.to_string(), kid);
        Some(ka)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::key_pool::selector::RoundRobinSelector;
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
    fn cu(uid: &str) -> RequestContext {
        RequestContext {
            user_id: Some(uid.to_string()),
            ..Default::default()
        }
    }
    #[test]
    fn test_sticky_same_user_gets_same_key() {
        let pool = vec![mk("k1"), mk("k2"), mk("k3")];
        let sel = StickySelector::new(Box::new(RoundRobinSelector::new()));
        let ctx = cu("alice");
        let first = sel.select(&pool, &ctx).unwrap().read().unwrap().id.clone();
        for _ in 0..9 {
            assert_eq!(
                sel.select(&pool, &ctx).unwrap().read().unwrap().id,
                first,
                "sticky: same user should always get the same key"
            );
        }
    }
    #[test]
    fn test_sticky_different_users_may_get_different_keys() {
        let pool = vec![mk("k1"), mk("k2"), mk("k3")];
        let sel = StickySelector::new(Box::new(RoundRobinSelector::new()));
        assert_eq!(
            sel.select(&pool, &cu("alice")).unwrap().read().unwrap().id,
            "k1"
        );
        assert_eq!(
            sel.select(&pool, &cu("bob")).unwrap().read().unwrap().id,
            "k2"
        );
        assert_eq!(
            sel.select(&pool, &cu("carol")).unwrap().read().unwrap().id,
            "k3"
        );
    }
    #[test]
    fn test_sticky_fallback_on_stale_key() {
        let pool = vec![mk("k1"), mk("k2")];
        let sel = StickySelector::new(Box::new(RoundRobinSelector::new()));
        let ctx = cu("alice");
        assert_eq!(sel.select(&pool, &ctx).unwrap().read().unwrap().id, "k1");
        pool[0].write().unwrap().health.status = KeyStatus::Disabled;
        assert_eq!(sel.select(&pool, &ctx).unwrap().read().unwrap().id, "k2");
    }
    #[test]
    fn test_sticky_no_user_falls_back_to_inner() {
        let pool = vec![mk("k1"), mk("k2")];
        assert!(
            StickySelector::new(Box::new(RoundRobinSelector::new()))
                .select(&pool, &RequestContext::default())
                .is_some()
        );
    }
    #[test]
    fn test_sticky_empty_pool_returns_none() {
        assert!(
            StickySelector::new(Box::new(RoundRobinSelector::new()))
                .select(&[], &cu("alice"))
                .is_none()
        );
    }
}
