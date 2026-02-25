//! Tower middleware stack for switchboard-server.
//!
//! # Layers (innermost → outermost, i.e. applied in reverse order)
//!
//! 1. [`request_id::RequestIdLayer`] — generates / propagates
//!    `X-Switchboard-Request-Id` on every request.
//! 2. [`auth_layer::AuthLayer`] — validates `Authorization` header via
//!    [`crate::auth::registry::AuthRegistry`]; injects [`crate::auth::validator::ValidatedClient`]
//!    into request extensions.  Returns 401 on failure.
//! 3. [`rate_limit::RateLimitLayer`] — token-bucket rate limiter per user.
//!    Returns 429 when limits are exceeded.
//!
//! # Usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use tower::ServiceBuilder;
//! use switchboard_server::middleware::{build_middleware_stack, MiddlewareConfig};
//! use switchboard_server::auth::registry::AuthRegistry;
//!
//! let registry = Arc::new(AuthRegistry::new(vec![]));
//! let stack = build_middleware_stack(MiddlewareConfig {
//!     auth_registry: registry,
//!     default_rpm: 60,
//!     default_tpm: 100_000,
//!     rate_limit_overrides: vec![],
//! });
//! ```

pub mod auth_layer;
pub mod rate_limit;
pub mod request_id;

pub use auth_layer::{AuthLayer, AuthService};
pub use rate_limit::{RateLimitLayer, RateLimitService, RateLimitSettings};
pub use request_id::{RequestId, RequestIdLayer, RequestIdService};

use std::sync::Arc;

use tower::ServiceBuilder;

use crate::auth::registry::AuthRegistry;

// ── Middleware configuration ───────────────────────────────────────────────────

/// Configuration for [`build_middleware_stack`].
#[derive(Debug)]
pub struct MiddlewareConfig {
    /// Auth registry used by the auth layer.
    pub auth_registry: Arc<AuthRegistry>,
    /// Default requests-per-minute for users without an override.
    pub default_rpm: u32,
    /// Default tokens-per-minute for users without an override.
    pub default_tpm: u32,
    /// Per-user rate limit overrides.
    pub rate_limit_overrides: Vec<(String, RateLimitSettings)>,
}

// ── Stack type alias ──────────────────────────────────────────────────────────

/// The concrete type of the full middleware stack produced by
/// [`build_middleware_stack`].
pub type MiddlewareStack = tower::layer::util::Stack<
    RateLimitLayer,
    tower::layer::util::Stack<
        AuthLayer,
        tower::layer::util::Stack<RequestIdLayer, tower::layer::util::Identity>,
    >,
>;

// ── Stack builder ─────────────────────────────────────────────────────────────

/// Build the standard switchboard-server Tower middleware stack.
///
/// Returns a [`ServiceBuilder`] with the following layers applied (outermost
/// first, i.e. the order in which a request passes through):
///
/// 1. `RequestIdLayer` — assign / propagate request ID
/// 2. `AuthLayer` — validate client credentials
/// 3. `RateLimitLayer` — enforce per-user rate limits
///
/// Callers wrap their axum `Router` with this builder:
///
/// ```rust,ignore
/// let app = stack.service(router);
/// ```
pub fn build_middleware_stack(cfg: MiddlewareConfig) -> ServiceBuilder<MiddlewareStack> {
    ServiceBuilder::new()
        .layer(RequestIdLayer)
        .layer(AuthLayer::new(cfg.auth_registry))
        .layer(RateLimitLayer::new(
            cfg.default_rpm,
            cfg.default_tpm,
            cfg.rate_limit_overrides,
        ))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::auth::registry::AuthRegistry;

    #[test]
    fn test_middleware_config_debug() {
        let cfg = MiddlewareConfig {
            auth_registry: Arc::new(AuthRegistry::new(vec![])),
            default_rpm: 60,
            default_tpm: 100_000,
            rate_limit_overrides: vec![],
        };
        // Verify Debug is implemented.
        let s = format!("{cfg:?}");
        assert!(s.contains("MiddlewareConfig"));
    }

    #[test]
    fn test_build_middleware_stack_compiles() {
        let cfg = MiddlewareConfig {
            auth_registry: Arc::new(AuthRegistry::new(vec![])),
            default_rpm: 60,
            default_tpm: 100_000,
            rate_limit_overrides: vec![(
                "power-user".to_owned(),
                RateLimitSettings {
                    rpm: 600,
                    tpm: 1_000_000,
                },
            )],
        };
        // Simply calling build is sufficient to prove the type-level stack compiles.
        let _stack = build_middleware_stack(cfg);
    }

    #[test]
    fn test_rate_limit_settings_fields() {
        let s = RateLimitSettings {
            rpm: 30,
            tpm: 50_000,
        };
        assert_eq!(s.rpm, 30);
        assert_eq!(s.tpm, 50_000);
    }
}
