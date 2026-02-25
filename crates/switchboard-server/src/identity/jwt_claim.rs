//! JWT-claim–based identity resolver.
//!
//! The auth middleware decodes the incoming JWT and injects each claim into
//! `RequestContext::switchboard_headers` under the key
//! `__jwt_claim_<claim_name>` (all lower-case).
//!
//! `JwtClaimResolver` is constructed with a claim name (e.g. `"email"`) and
//! reads the corresponding header entry.

use async_trait::async_trait;
use switchboard_common::types::RequestContext;

use crate::identity::resolver::{IdentityResolver, IdentitySource, UserIdentity};

/// Header-key prefix used by the auth middleware when injecting JWT claims.
pub const JWT_CLAIM_PREFIX: &str = "__jwt_claim_";

/// Resolves user identity from a named JWT claim.
///
/// The claim value is read from `RequestContext::switchboard_headers` under the
/// key `__jwt_claim_<claim_name>`.  The auth middleware is responsible for
/// populating these entries after validating the JWT.
#[derive(Debug)]
pub struct JwtClaimResolver {
    /// The JWT claim name to extract (e.g. `"email"`, `"sub"`, `"username"`).
    pub claim_name: String,
}

impl JwtClaimResolver {
    pub fn new(claim_name: impl Into<String>) -> Self {
        Self {
            claim_name: claim_name.into(),
        }
    }

    fn header_key(&self) -> String {
        format!("{JWT_CLAIM_PREFIX}{}", self.claim_name.to_lowercase())
    }
}

#[async_trait]
impl IdentityResolver for JwtClaimResolver {
    async fn resolve(&self, ctx: &RequestContext) -> Option<UserIdentity> {
        let key = self.header_key();
        let value = ctx.switchboard_headers.get(&key)?;
        if value.is_empty() {
            return None;
        }
        Some(UserIdentity {
            id: value.clone(),
            name: None,
            team: None,
            source: IdentitySource::JwtClaim(self.claim_name.clone()),
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with_claim(claim: &str, value: &str) -> RequestContext {
        let mut ctx = RequestContext::default();
        let key = format!("{JWT_CLAIM_PREFIX}{}", claim.to_lowercase());
        ctx.switchboard_headers.insert(key, value.to_string());
        ctx
    }

    #[tokio::test]
    async fn test_jwt_claim_resolver_returns_identity_when_claim_present() {
        let ctx = ctx_with_claim("email", "alice@example.com");
        let resolver = JwtClaimResolver::new("email");
        let id = resolver.resolve(&ctx).await.unwrap();
        assert_eq!(id.id, "alice@example.com");
        assert_eq!(id.source, IdentitySource::JwtClaim("email".into()));
        assert!(id.name.is_none());
        assert!(id.team.is_none());
    }

    #[tokio::test]
    async fn test_jwt_claim_resolver_case_insensitive_claim_name() {
        // Claim stored with lowercase key; resolver uses uppercase name.
        let ctx = ctx_with_claim("sub", "user-123");
        let resolver = JwtClaimResolver::new("SUB");
        let id = resolver.resolve(&ctx).await.unwrap();
        assert_eq!(id.id, "user-123");
        assert_eq!(id.source, IdentitySource::JwtClaim("SUB".into()));
    }

    #[tokio::test]
    async fn test_jwt_claim_resolver_returns_none_when_claim_absent() {
        let ctx = RequestContext::default();
        let resolver = JwtClaimResolver::new("email");
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_jwt_claim_resolver_returns_none_for_empty_value() {
        let ctx = ctx_with_claim("email", "");
        let resolver = JwtClaimResolver::new("email");
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_jwt_claim_resolver_wrong_claim_returns_none() {
        let ctx = ctx_with_claim("email", "alice@example.com");
        let resolver = JwtClaimResolver::new("username");
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[test]
    fn test_jwt_claim_resolver_header_key_format() {
        let r = JwtClaimResolver::new("Email");
        assert_eq!(r.header_key(), "__jwt_claim_email");
    }
}
