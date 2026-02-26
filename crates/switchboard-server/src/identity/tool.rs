//! Tool-specific identity resolver.
//!
//! [`ToolSpecificResolver`] detects the AI coding tool that made the request
//! from standard HTTP headers and enriches the [`UserIdentity`] with the tool
//! name via [`IdentitySource::ToolSpecific`].
//!
//! It does NOT override `user_id` — it only annotates the identity so that
//! the tool name appears in observability spans.
//!
//! Detection priority:
//! 1. `x-switchboard-tool` — explicit header set by `switchboard-local`
//! 2. `user-agent` — sniffed for common tool patterns

use std::collections::HashMap;

use async_trait::async_trait;
use switchboard_common::types::RequestContext;

use crate::identity::resolver::{IdentityResolver, IdentitySource, UserIdentity};

/// The Switchboard-protocol header that `switchboard-local` uses to identify
/// the client tool explicitly.
pub const HEADER_TOOL: &str = "x-switchboard-tool";

// ── detect_tool ───────────────────────────────────────────────────────────────

/// Detect the tool name from request headers.
///
/// Returns `None` if the tool cannot be determined.
///
/// # Detection order
/// 1. `x-switchboard-tool` header — used directly if present.
/// 2. `user-agent` header — matched against known tool patterns.
pub fn detect_tool(headers: &HashMap<String, String>) -> Option<String> {
    // 1. Explicit header wins over user-agent sniffing.
    if let Some(tool) = headers.get(HEADER_TOOL) {
        let trimmed = tool.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    // 2. User-agent sniffing.
    let ua = headers.get("user-agent")?;
    let ua_lower = ua.to_lowercase();

    if ua_lower.contains("claude-code") || ua_lower.contains("anthropic-claude") {
        Some("claude-code".to_string())
    } else if ua_lower.contains("cursor") {
        Some("cursor".to_string())
    } else if ua_lower.contains("aider") {
        Some("aider".to_string())
    } else if ua_lower.contains("continue") {
        Some("continue".to_string())
    } else if ua_lower.contains("opencode") {
        Some("opencode".to_string())
    } else {
        None
    }
}

// ── ToolSpecificResolver ──────────────────────────────────────────────────────

/// Resolver that detects the client tool from request headers and sets
/// `UserIdentity.source = IdentitySource::ToolSpecific(tool_name)`.
///
/// Does NOT override `user_id` — it only enriches the identity with the
/// tool name so it appears in observability spans.
#[derive(Debug, Default)]
pub struct ToolSpecificResolver;

#[async_trait]
impl IdentityResolver for ToolSpecificResolver {
    async fn resolve(&self, ctx: &RequestContext) -> Option<UserIdentity> {
        let tool_name = detect_tool(&ctx.switchboard_headers)?;

        tracing::debug!(
            tool = %tool_name,
            user_id = ?ctx.user_id,
            "ToolSpecificResolver: detected tool"
        );

        Some(UserIdentity {
            id: ctx
                .user_id
                .clone()
                .unwrap_or_else(|| "anonymous".to_string()),
            name: None,
            team: ctx.team.clone(),
            source: IdentitySource::ToolSpecific(tool_name),
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn ctx_with_headers(pairs: &[(&str, &str)]) -> RequestContext {
        RequestContext {
            switchboard_headers: headers(pairs),
            ..RequestContext::default()
        }
    }

    // ── detect_tool ───────────────────────────────────────────────────────────

    #[test]
    fn test_detect_tool_from_explicit_header() {
        let h = headers(&[(HEADER_TOOL, "cursor")]);
        assert_eq!(detect_tool(&h), Some("cursor".to_string()));
    }

    #[test]
    fn test_detect_tool_from_user_agent_claude_code() {
        let h = headers(&[("user-agent", "claude-code/1.0")]);
        assert_eq!(detect_tool(&h), Some("claude-code".to_string()));
    }

    #[test]
    fn test_detect_tool_from_user_agent_cursor() {
        let h = headers(&[("user-agent", "Cursor/0.42")]);
        assert_eq!(detect_tool(&h), Some("cursor".to_string()));
    }

    #[test]
    fn test_detect_tool_from_user_agent_aider() {
        let h = headers(&[("user-agent", "aider/0.50")]);
        assert_eq!(detect_tool(&h), Some("aider".to_string()));
    }

    #[test]
    fn test_detect_tool_from_user_agent_continue() {
        let h = headers(&[("user-agent", "continue/1.2.3")]);
        assert_eq!(detect_tool(&h), Some("continue".to_string()));
    }

    #[test]
    fn test_detect_tool_from_user_agent_opencode() {
        let h = headers(&[("user-agent", "opencode/0.1.0")]);
        assert_eq!(detect_tool(&h), Some("opencode".to_string()));
    }

    #[test]
    fn test_detect_tool_from_user_agent_anthropic_claude() {
        let h = headers(&[("user-agent", "anthropic-claude/1.0")]);
        assert_eq!(detect_tool(&h), Some("claude-code".to_string()));
    }

    #[test]
    fn test_detect_tool_unknown_ua() {
        let h = headers(&[("user-agent", "Mozilla/5.0")]);
        assert_eq!(detect_tool(&h), None);
    }

    #[test]
    fn test_detect_tool_no_headers() {
        let h = HashMap::new();
        assert_eq!(detect_tool(&h), None);
    }

    #[test]
    fn test_detect_tool_explicit_header_wins_over_ua() {
        // Both x-switchboard-tool and user-agent present — explicit header wins.
        let h = headers(&[(HEADER_TOOL, "aider"), ("user-agent", "cursor/0.42")]);
        assert_eq!(
            detect_tool(&h),
            Some("aider".to_string()),
            "explicit x-switchboard-tool should win over user-agent"
        );
    }

    // ── ToolSpecificResolver ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_tool_resolver_returns_none_when_undetected() {
        let ctx = RequestContext::default(); // no headers
        let resolver = ToolSpecificResolver;
        assert!(resolver.resolve(&ctx).await.is_none());
    }

    #[tokio::test]
    async fn test_tool_resolver_preserves_user_id() {
        let mut ctx = ctx_with_headers(&[(HEADER_TOOL, "cursor")]);
        ctx.user_id = Some("alice@example.com".into());
        ctx.team = Some("platform".into());

        let resolver = ToolSpecificResolver;
        let identity = resolver.resolve(&ctx).await.unwrap();

        assert_eq!(
            identity.id, "alice@example.com",
            "resolver should preserve existing user_id"
        );
        assert_eq!(
            identity.team.as_deref(),
            Some("platform"),
            "resolver should preserve existing team"
        );
    }

    #[tokio::test]
    async fn test_tool_resolver_sets_source_tool_specific() {
        let ctx = ctx_with_headers(&[(HEADER_TOOL, "cursor")]);
        let resolver = ToolSpecificResolver;
        let identity = resolver.resolve(&ctx).await.unwrap();

        assert_eq!(
            identity.source,
            IdentitySource::ToolSpecific("cursor".to_string()),
            "source should be ToolSpecific with the detected tool name"
        );
    }

    #[tokio::test]
    async fn test_tool_resolver_anonymous_when_no_user_id() {
        // When context has no user_id, the resolver falls back to "anonymous".
        let ctx = ctx_with_headers(&[(HEADER_TOOL, "aider")]);
        let resolver = ToolSpecificResolver;
        let identity = resolver.resolve(&ctx).await.unwrap();

        assert_eq!(identity.id, "anonymous");
    }

    #[tokio::test]
    async fn test_tool_resolver_ua_sniff_claude_code() {
        let ctx = ctx_with_headers(&[("user-agent", "claude-code/1.2.3")]);
        let resolver = ToolSpecificResolver;
        let identity = resolver.resolve(&ctx).await.unwrap();

        assert_eq!(
            identity.source,
            IdentitySource::ToolSpecific("claude-code".to_string())
        );
    }
}
