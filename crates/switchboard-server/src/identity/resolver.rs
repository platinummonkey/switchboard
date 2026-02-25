//! User identity types and the `IdentityResolver` trait.

use async_trait::async_trait;
use switchboard_common::types::RequestContext;

// ── Identity source ───────────────────────────────────────────────────────────

/// How a user's identity was determined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentitySource {
    /// Extracted from the `X-Switchboard-User` header.
    Header,
    /// Extracted from a JWT claim (contains the claim name).
    JwtClaim(String),
    /// Looked up via an API-key → user mapping table.
    ApiKeyMapping,
    /// Extracted from the mTLS client certificate Common Name.
    MtlsCn,
    /// Provided by a tool-specific header (contains the tool name).
    ToolSpecific(String),
    /// No identity could be determined; request is treated as anonymous.
    Anonymous,
}

impl std::fmt::Display for IdentitySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdentitySource::Header => write!(f, "header"),
            IdentitySource::JwtClaim(c) => write!(f, "jwt_claim:{c}"),
            IdentitySource::ApiKeyMapping => write!(f, "api_key_mapping"),
            IdentitySource::MtlsCn => write!(f, "mtls_cn"),
            IdentitySource::ToolSpecific(t) => write!(f, "tool:{t}"),
            IdentitySource::Anonymous => write!(f, "anonymous"),
        }
    }
}

// ── User identity ─────────────────────────────────────────────────────────────

/// Resolved user identity attached to every trace span and metric.
#[derive(Debug, Clone)]
pub struct UserIdentity {
    /// Primary identifier — email, username, or opaque ID.
    pub id: String,
    /// Optional human-readable display name.
    pub name: Option<String>,
    /// Optional team or group.
    pub team: Option<String>,
    /// How this identity was resolved.
    pub source: IdentitySource,
}

impl UserIdentity {
    pub fn anonymous() -> Self {
        Self {
            id: "anonymous".into(),
            name: None,
            team: None,
            source: IdentitySource::Anonymous,
        }
    }

    pub fn is_anonymous(&self) -> bool {
        matches!(self.source, IdentitySource::Anonymous)
    }
}

impl std::fmt::Display for UserIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}

// ── Resolver trait ────────────────────────────────────────────────────────────

/// A single link in the identity resolution chain.
///
/// Implementations are tried in order; the first one that returns `Some` wins.
/// Built-in implementations: `HeaderResolver`, `JwtClaimResolver`,
/// `ApiKeyMappingResolver`, `MtlsCnResolver` (Phase 6).
#[async_trait]
pub trait IdentityResolver: Send + Sync {
    /// Attempt to extract a [`UserIdentity`] from the request context.
    /// Returns `None` to pass to the next resolver in the chain.
    async fn resolve(&self, ctx: &RequestContext) -> Option<UserIdentity>;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `IdentityResolver` is object-safe.
    fn _assert_object_safe(_: &dyn IdentityResolver) {}

    #[test]
    fn test_anonymous_identity() {
        let id = UserIdentity::anonymous();
        assert!(id.is_anonymous());
        assert_eq!(id.id, "anonymous");
        assert!(id.name.is_none());
    }

    #[test]
    fn test_named_identity_not_anonymous() {
        let id = UserIdentity {
            id: "alice@example.com".into(),
            name: Some("Alice".into()),
            team: Some("platform".into()),
            source: IdentitySource::Header,
        };
        assert!(!id.is_anonymous());
        assert_eq!(id.to_string(), "alice@example.com");
    }

    #[test]
    fn test_source_display() {
        assert_eq!(IdentitySource::Header.to_string(), "header");
        assert_eq!(
            IdentitySource::JwtClaim("email".into()).to_string(),
            "jwt_claim:email"
        );
        assert_eq!(IdentitySource::ApiKeyMapping.to_string(), "api_key_mapping");
        assert_eq!(IdentitySource::MtlsCn.to_string(), "mtls_cn");
        assert_eq!(
            IdentitySource::ToolSpecific("cursor".into()).to_string(),
            "tool:cursor"
        );
        assert_eq!(IdentitySource::Anonymous.to_string(), "anonymous");
    }

    #[tokio::test]
    async fn test_resolver_returns_none_by_default() {
        struct NullResolver;
        #[async_trait::async_trait]
        impl IdentityResolver for NullResolver {
            async fn resolve(&self, _ctx: &RequestContext) -> Option<UserIdentity> {
                None
            }
        }
        let r = NullResolver;
        let ctx = RequestContext::default();
        assert!(r.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_header_resolver_stub() {
        struct HeaderResolver;
        #[async_trait::async_trait]
        impl IdentityResolver for HeaderResolver {
            async fn resolve(&self, ctx: &RequestContext) -> Option<UserIdentity> {
                ctx.user_id.as_ref().map(|id| UserIdentity {
                    id: id.clone(),
                    name: None,
                    team: ctx.team.clone(),
                    source: IdentitySource::Header,
                })
            }
        }
        let mut ctx = RequestContext::new();
        ctx.user_id = Some("bob@example.com".into());
        ctx.team = Some("infra".into());
        let id = HeaderResolver.resolve(&ctx).await.unwrap();
        assert_eq!(id.id, "bob@example.com");
        assert_eq!(id.team.as_deref(), Some("infra"));
        assert_eq!(id.source, IdentitySource::Header);
    }
}
