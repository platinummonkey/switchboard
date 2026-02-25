//! mTLS Common Name identity resolver.
//!
//! The TLS termination layer (added in a later phase) will extract the
//! client certificate's Common Name and store it in
//! `RequestContext::switchboard_headers` under the key [`MTLS_CN_HEADER`].
//!
//! [`MtlsCnResolver`] reads that header and returns a [`UserIdentity`] whose
//! [`IdentitySource`] is [`IdentitySource::MtlsCn`].

use async_trait::async_trait;
use switchboard_common::types::RequestContext;

use crate::identity::resolver::{IdentityResolver, IdentitySource, UserIdentity};

/// The key under which the TLS layer stores the client certificate CN.
pub const MTLS_CN_HEADER: &str = "__mtls_cn";

// ── Extension type ────────────────────────────────────────────────────────────

/// Axum request extension populated by the TLS termination middleware.
///
/// Contains the Common Name extracted from the verified client certificate.
/// The TLS layer also copies this into `RequestContext::switchboard_headers`
/// under [`MTLS_CN_HEADER`] so that identity resolvers — which only receive a
/// [`RequestContext`] — can read it without needing direct access to axum
/// extensions.
#[derive(Debug, Clone)]
pub struct MtlsClientCn(pub String);

// ── MtlsCnResolver ────────────────────────────────────────────────────────────

/// Resolves user identity from the mTLS client certificate Common Name.
///
/// Reads the CN injected by the TLS layer from
/// `ctx.switchboard_headers["__mtls_cn"]`.  Returns `None` if the header is
/// absent or empty (i.e. the connection was not mutually authenticated).
#[derive(Debug, Default)]
pub struct MtlsCnResolver;

#[async_trait]
impl IdentityResolver for MtlsCnResolver {
    async fn resolve(&self, ctx: &RequestContext) -> Option<UserIdentity> {
        let cn = ctx.switchboard_headers.get(MTLS_CN_HEADER)?;
        if cn.is_empty() {
            return None;
        }
        Some(UserIdentity {
            id: cn.clone(),
            name: None,
            team: None,
            source: IdentitySource::MtlsCn,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with_cn(cn: &str) -> RequestContext {
        let mut ctx = RequestContext::default();
        ctx.switchboard_headers
            .insert(MTLS_CN_HEADER.to_string(), cn.to_string());
        ctx
    }

    #[tokio::test]
    async fn test_mtls_cn_resolver_returns_identity_when_cn_present() {
        let resolver = MtlsCnResolver;
        let ctx = ctx_with_cn("alice@example.com");
        let id = resolver.resolve(&ctx).await.unwrap();
        assert_eq!(id.id, "alice@example.com");
        assert_eq!(id.source, IdentitySource::MtlsCn);
        assert!(id.name.is_none());
        assert!(id.team.is_none());
    }

    #[tokio::test]
    async fn test_mtls_cn_resolver_returns_none_when_header_absent() {
        let resolver = MtlsCnResolver;
        let ctx = RequestContext::default();
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_mtls_cn_resolver_returns_none_for_empty_cn() {
        let resolver = MtlsCnResolver;
        let ctx = ctx_with_cn("");
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_mtls_cn_resolver_preserves_cn_exactly() {
        let resolver = MtlsCnResolver;
        let ctx = ctx_with_cn("CN=service-account,OU=platform,O=Datadog");
        let id = resolver.resolve(&ctx).await.unwrap();
        assert_eq!(id.id, "CN=service-account,OU=platform,O=Datadog");
    }

    #[tokio::test]
    async fn test_mtls_cn_resolver_source_is_mtls_cn() {
        let resolver = MtlsCnResolver;
        let ctx = ctx_with_cn("bot@ci.example.com");
        let id = resolver.resolve(&ctx).await.unwrap();
        assert!(matches!(id.source, IdentitySource::MtlsCn));
    }

    #[test]
    fn test_mtls_client_cn_extension_debug() {
        let cn = MtlsClientCn("test.example.com".into());
        let debug_str = format!("{cn:?}");
        assert!(debug_str.contains("test.example.com"));
    }

    #[test]
    fn test_mtls_client_cn_clone() {
        let cn = MtlsClientCn("clone-me".into());
        let cn2 = cn.clone();
        assert_eq!(cn.0, cn2.0);
    }
}
