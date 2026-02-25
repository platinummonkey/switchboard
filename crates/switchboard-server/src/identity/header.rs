//! Header-based identity resolver.
//!
//! Reads `X-Switchboard-User` and `X-Switchboard-Team` from the request's
//! Switchboard protocol headers.

use async_trait::async_trait;
use switchboard_common::types::RequestContext;

use crate::identity::resolver::{IdentityResolver, IdentitySource, UserIdentity};

/// The Switchboard-protocol header key for the user identifier.
pub const HEADER_USER: &str = "x-switchboard-user";
/// The Switchboard-protocol header key for the team identifier.
pub const HEADER_TEAM: &str = "x-switchboard-team";

/// Resolves identity from Switchboard-specific HTTP headers.
///
/// Returns `Some` if `X-Switchboard-User` is present and non-empty.
/// Optionally also reads `X-Switchboard-Team`.
#[derive(Debug, Default)]
pub struct HeaderResolver;

#[async_trait]
impl IdentityResolver for HeaderResolver {
    async fn resolve(&self, ctx: &RequestContext) -> Option<UserIdentity> {
        let user_id = ctx.switchboard_headers.get(HEADER_USER)?;
        if user_id.is_empty() {
            return None;
        }
        let team = ctx.switchboard_headers.get(HEADER_TEAM).cloned();
        Some(UserIdentity {
            id: user_id.clone(),
            name: None,
            team,
            source: IdentitySource::Header,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with_headers(pairs: &[(&str, &str)]) -> RequestContext {
        let mut ctx = RequestContext::default();
        for (k, v) in pairs {
            ctx.switchboard_headers.insert(k.to_string(), v.to_string());
        }
        ctx
    }

    #[tokio::test]
    async fn test_header_resolver_returns_identity_when_user_present() {
        let ctx = ctx_with_headers(&[
            (HEADER_USER, "alice@example.com"),
            (HEADER_TEAM, "platform"),
        ]);
        let resolver = HeaderResolver;
        let id = resolver.resolve(&ctx).await.unwrap();
        assert_eq!(id.id, "alice@example.com");
        assert_eq!(id.team.as_deref(), Some("platform"));
        assert_eq!(id.source, IdentitySource::Header);
        assert!(id.name.is_none());
    }

    #[tokio::test]
    async fn test_header_resolver_returns_none_when_user_absent() {
        let ctx = ctx_with_headers(&[(HEADER_TEAM, "platform")]);
        let resolver = HeaderResolver;
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_header_resolver_returns_none_for_empty_user() {
        let ctx = ctx_with_headers(&[(HEADER_USER, "")]);
        let resolver = HeaderResolver;
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_header_resolver_team_optional() {
        let ctx = ctx_with_headers(&[(HEADER_USER, "bob@example.com")]);
        let resolver = HeaderResolver;
        let id = resolver.resolve(&ctx).await.unwrap();
        assert_eq!(id.id, "bob@example.com");
        assert!(id.team.is_none());
    }

    #[tokio::test]
    async fn test_header_resolver_empty_context_returns_none() {
        let ctx = RequestContext::default();
        let resolver = HeaderResolver;
        assert!(resolver.resolve(&ctx).await.is_none());
    }
}
