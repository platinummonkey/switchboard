//! `KeySelector` trait and `KeyPool` skeleton.
//!
//! Concrete selector implementations (WeightedRandom, RoundRobin, etc.)
//! live in Phase 5 (`key_pool/selector/`).

use switchboard_common::types::RequestContext;

use crate::key_pool::PooledKey;

// ── Selector trait ────────────────────────────────────────────────────────────

/// Chooses a key from the pool for a given request.
///
/// Implementations must be `Send + Sync` so they can be shared across async
/// tasks. Selection must be synchronous and fast (no I/O).
pub trait KeySelector: Send + Sync {
    /// Return a reference to the chosen key, or `None` if the pool is empty
    /// or all keys are disabled.
    fn select<'a>(&self, pool: &'a [PooledKey], request: &RequestContext) -> Option<&'a PooledKey>;
}

// ── KeyPool skeleton ──────────────────────────────────────────────────────────

/// Runtime key pool for one upstream provider.
///
/// Holds an ordered list of [`PooledKey`] entries and delegates selection to
/// a pluggable [`KeySelector`].  Health updates and rotation algorithms are
/// implemented in Phase 5.
pub struct KeyPool {
    pub(crate) keys: Vec<PooledKey>,
    pub(crate) selector: Box<dyn KeySelector>,
}

impl KeyPool {
    /// Construct a pool from a pre-built key list and selector.
    pub fn new(keys: Vec<PooledKey>, selector: Box<dyn KeySelector>) -> Self {
        Self { keys, selector }
    }

    /// Number of keys in the pool (all statuses).
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// `true` if the pool contains no keys at all.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Number of keys currently eligible to serve requests.
    pub fn eligible_count(&self) -> usize {
        self.keys.iter().filter(|k| k.is_eligible()).count()
    }

    /// Select a key for the given request context.
    pub fn select(&self, request: &RequestContext) -> Option<&PooledKey> {
        self.selector.select(&self.keys, request)
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

    /// Stub selector that always returns the first eligible key.
    struct FirstEligible;
    impl KeySelector for FirstEligible {
        fn select<'a>(
            &self,
            pool: &'a [PooledKey],
            _req: &RequestContext,
        ) -> Option<&'a PooledKey> {
            pool.iter().find(|k| k.is_eligible())
        }
    }

    fn pool(keys: Vec<PooledKey>) -> KeyPool {
        KeyPool::new(keys, Box::new(FirstEligible))
    }

    /// Verifies `KeySelector` is object-safe.
    fn _assert_selector_object_safe(_: &dyn KeySelector) {}

    #[test]
    fn test_pool_len_and_empty() {
        let p = pool(vec![make_key("k1"), make_key("k2")]);
        assert_eq!(p.len(), 2);
        assert!(!p.is_empty());
        assert_eq!(pool(vec![]).len(), 0);
        assert!(pool(vec![]).is_empty());
    }

    #[test]
    fn test_eligible_count_excludes_disabled() {
        let mut keys = vec![make_key("k1"), make_key("k2"), make_key("k3")];
        keys[1].health.status = KeyStatus::Disabled;
        let p = pool(keys);
        assert_eq!(p.eligible_count(), 2);
    }

    #[test]
    fn test_select_returns_first_eligible() {
        let mut keys = vec![make_key("k1"), make_key("k2")];
        keys[0].health.status = KeyStatus::Disabled;
        let p = pool(keys);
        let ctx = RequestContext::default();
        let selected = p.select(&ctx).unwrap();
        assert_eq!(selected.id, "k2");
    }

    #[test]
    fn test_select_none_when_all_disabled() {
        let mut keys = vec![make_key("k1")];
        keys[0].health.status = KeyStatus::Disabled;
        let p = pool(keys);
        let ctx = RequestContext::default();
        assert!(p.select(&ctx).is_none());
    }

    #[test]
    fn test_select_none_on_empty_pool() {
        let p = pool(vec![]);
        let ctx = RequestContext::default();
        assert!(p.select(&ctx).is_none());
    }
}
