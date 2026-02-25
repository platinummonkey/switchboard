//! API-key–to–identity mapping resolver.
//!
//! The auth middleware extracts the raw API key from the `Authorization` header
//! and injects it into `RequestContext::switchboard_headers` under the key
//! `__api_key`.  This resolver looks up that value in a static mapping table.

use std::collections::HashMap;

use async_trait::async_trait;
use switchboard_common::types::RequestContext;

use crate::identity::resolver::{IdentityResolver, IdentitySource, UserIdentity};

/// Header key under which the auth middleware stores the raw API key.
pub const API_KEY_HEADER: &str = "__api_key";

/// Resolves user identity by looking up the request's API key in a mapping
/// table provided at construction time.
///
/// The mapping is a `HashMap<String, UserIdentity>` where the key is the
/// plaintext API key and the value is the pre-configured identity.  A copy
/// of the mapped identity is returned with `source` set to
/// [`IdentitySource::ApiKeyMapping`].
#[derive(Debug)]
pub struct ApiKeyMappingResolver {
    mapping: HashMap<String, UserIdentity>,
}

impl ApiKeyMappingResolver {
    /// Construct the resolver from an `api_key → UserIdentity` map.
    pub fn new(mapping: HashMap<String, UserIdentity>) -> Self {
        Self { mapping }
    }
}

#[async_trait]
impl IdentityResolver for ApiKeyMappingResolver {
    async fn resolve(&self, ctx: &RequestContext) -> Option<UserIdentity> {
        let api_key = ctx.switchboard_headers.get(API_KEY_HEADER)?;
        let mut identity = self.mapping.get(api_key)?.clone();
        // Always stamp the source so callers know how identity was determined.
        identity.source = IdentitySource::ApiKeyMapping;
        Some(identity)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mapping() -> HashMap<String, UserIdentity> {
        let mut m = HashMap::new();
        m.insert(
            "sk-alice-123".to_string(),
            UserIdentity {
                id: "alice@example.com".to_string(),
                name: Some("Alice".to_string()),
                team: Some("platform".to_string()),
                source: IdentitySource::Header, // will be overwritten
            },
        );
        m.insert(
            "sk-bob-456".to_string(),
            UserIdentity {
                id: "bob@example.com".to_string(),
                name: None,
                team: None,
                source: IdentitySource::Header,
            },
        );
        m
    }

    fn ctx_with_api_key(key: &str) -> RequestContext {
        let mut ctx = RequestContext::default();
        ctx.switchboard_headers
            .insert(API_KEY_HEADER.to_string(), key.to_string());
        ctx
    }

    #[tokio::test]
    async fn test_api_key_resolver_returns_identity_for_known_key() {
        let resolver = ApiKeyMappingResolver::new(make_mapping());
        let ctx = ctx_with_api_key("sk-alice-123");
        let id = resolver.resolve(&ctx).await.unwrap();
        assert_eq!(id.id, "alice@example.com");
        assert_eq!(id.name.as_deref(), Some("Alice"));
        assert_eq!(id.team.as_deref(), Some("platform"));
        // Source is always overwritten to ApiKeyMapping.
        assert_eq!(id.source, IdentitySource::ApiKeyMapping);
    }

    #[tokio::test]
    async fn test_api_key_resolver_second_key() {
        let resolver = ApiKeyMappingResolver::new(make_mapping());
        let ctx = ctx_with_api_key("sk-bob-456");
        let id = resolver.resolve(&ctx).await.unwrap();
        assert_eq!(id.id, "bob@example.com");
        assert_eq!(id.source, IdentitySource::ApiKeyMapping);
    }

    #[tokio::test]
    async fn test_api_key_resolver_returns_none_for_unknown_key() {
        let resolver = ApiKeyMappingResolver::new(make_mapping());
        let ctx = ctx_with_api_key("sk-unknown-999");
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_api_key_resolver_returns_none_when_header_absent() {
        let resolver = ApiKeyMappingResolver::new(make_mapping());
        let ctx = RequestContext::default();
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_api_key_resolver_empty_mapping_always_returns_none() {
        let resolver = ApiKeyMappingResolver::new(HashMap::new());
        let ctx = ctx_with_api_key("sk-alice-123");
        assert!(resolver.resolve(&ctx).await.is_none());
    }
}
