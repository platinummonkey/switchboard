//! Identity resolution chain.
//!
//! [`IdentityChain`] tries each resolver in order and returns the first
//! non-`None` result.  If all resolvers return `None` it falls back to
//! [`UserIdentity::anonymous()`].

use async_trait::async_trait;
use switchboard_common::types::RequestContext;

use crate::identity::resolver::{IdentityResolver, UserIdentity};

/// Ordered chain of identity resolvers.
///
/// Call [`IdentityChain::resolve`] to run each resolver in order.  The first
/// resolver that returns `Some(identity)` wins.  If all resolvers return
/// `None`, an anonymous identity is returned.
pub struct IdentityChain {
    resolvers: Vec<Box<dyn IdentityResolver>>,
}

impl std::fmt::Debug for IdentityChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdentityChain")
            .field("resolver_count", &self.resolvers.len())
            .finish()
    }
}

impl IdentityChain {
    /// Build a chain from an ordered list of resolvers.
    pub fn new(resolvers: Vec<Box<dyn IdentityResolver>>) -> Self {
        Self { resolvers }
    }

    /// Try each resolver in order and return the first match.
    ///
    /// Falls back to [`UserIdentity::anonymous()`] if no resolver matches.
    pub async fn resolve(&self, ctx: &RequestContext) -> UserIdentity {
        for resolver in &self.resolvers {
            if let Some(identity) = resolver.resolve(ctx).await {
                return identity;
            }
        }
        UserIdentity::anonymous()
    }
}

// ── IdentityResolver impl for IdentityChain (composability) ──────────────────

#[async_trait]
impl IdentityResolver for IdentityChain {
    async fn resolve(&self, ctx: &RequestContext) -> Option<UserIdentity> {
        for resolver in &self.resolvers {
            if let Some(identity) = resolver.resolve(ctx).await {
                return Some(identity);
            }
        }
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::identity::resolver::{IdentitySource, UserIdentity};

    // ── Test helpers ──────────────────────────────────────────────────────────

    struct AlwaysNoneResolver;
    #[async_trait]
    impl IdentityResolver for AlwaysNoneResolver {
        async fn resolve(&self, _ctx: &RequestContext) -> Option<UserIdentity> {
            None
        }
    }

    struct FixedResolver(UserIdentity);
    #[async_trait]
    impl IdentityResolver for FixedResolver {
        async fn resolve(&self, _ctx: &RequestContext) -> Option<UserIdentity> {
            Some(self.0.clone())
        }
    }

    fn alice() -> UserIdentity {
        UserIdentity {
            id: "alice@example.com".into(),
            name: Some("Alice".into()),
            team: Some("platform".into()),
            source: IdentitySource::Header,
        }
    }

    fn bob() -> UserIdentity {
        UserIdentity {
            id: "bob@example.com".into(),
            name: None,
            team: None,
            source: IdentitySource::JwtClaim("email".into()),
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_chain_all_none_returns_anonymous() {
        let chain = IdentityChain::new(vec![
            Box::new(AlwaysNoneResolver),
            Box::new(AlwaysNoneResolver),
        ]);
        let ctx = RequestContext::default();
        let id = chain.resolve(&ctx).await;
        assert!(id.is_anonymous());
        assert_eq!(id.id, "anonymous");
    }

    #[tokio::test]
    async fn test_chain_empty_returns_anonymous() {
        let chain = IdentityChain::new(vec![]);
        let ctx = RequestContext::default();
        let id = chain.resolve(&ctx).await;
        assert!(id.is_anonymous());
    }

    #[tokio::test]
    async fn test_chain_first_resolver_matches_returns_immediately() {
        let chain = IdentityChain::new(vec![
            Box::new(FixedResolver(alice())),
            Box::new(FixedResolver(bob())), // should never be reached
        ]);
        let ctx = RequestContext::default();
        let id = chain.resolve(&ctx).await;
        assert_eq!(id.id, "alice@example.com");
        assert_eq!(id.source, IdentitySource::Header);
    }

    #[tokio::test]
    async fn test_chain_second_resolver_matches_when_first_none() {
        let chain = IdentityChain::new(vec![
            Box::new(AlwaysNoneResolver),
            Box::new(FixedResolver(bob())),
        ]);
        let ctx = RequestContext::default();
        let id = chain.resolve(&ctx).await;
        assert_eq!(id.id, "bob@example.com");
        assert_eq!(id.source, IdentitySource::JwtClaim("email".into()));
    }

    #[tokio::test]
    async fn test_chain_trait_impl_returns_none_when_all_none() {
        // When used as an IdentityResolver itself, the chain should return None
        // (not anonymous) if no resolver matches — composability.
        let chain = IdentityChain::new(vec![Box::new(AlwaysNoneResolver)]);
        let ctx = RequestContext::default();
        let result: Option<UserIdentity> =
            <IdentityChain as IdentityResolver>::resolve(&chain, &ctx).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_chain_trait_impl_returns_some_when_match() {
        let chain = IdentityChain::new(vec![
            Box::new(AlwaysNoneResolver),
            Box::new(FixedResolver(alice())),
        ]);
        let ctx = RequestContext::default();
        let result: Option<UserIdentity> =
            <IdentityChain as IdentityResolver>::resolve(&chain, &ctx).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, "alice@example.com");
    }

    #[tokio::test]
    async fn test_chain_with_real_resolvers() {
        use crate::identity::header::{HEADER_USER, HeaderResolver};

        let mut ctx = RequestContext::default();
        ctx.switchboard_headers
            .insert(HEADER_USER.to_string(), "carol@example.com".to_string());

        let chain =
            IdentityChain::new(vec![Box::new(AlwaysNoneResolver), Box::new(HeaderResolver)]);
        let id = chain.resolve(&ctx).await;
        assert_eq!(id.id, "carol@example.com");
        assert_eq!(id.source, IdentitySource::Header);
    }
}
