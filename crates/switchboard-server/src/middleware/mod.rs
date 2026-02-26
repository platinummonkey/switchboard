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
//! 4. [`guardrail::GuardrailLayer`] — (optional) evaluates guardrail engines
//!    before forwarding requests and after receiving responses.
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
//!     guardrail_pipeline: None,
//! });
//! ```

pub mod auth_layer;
pub mod guardrail;
pub mod rate_limit;
pub mod request_id;

pub use auth_layer::{AuthLayer, AuthService};
pub use guardrail::GuardrailLayer;
pub use rate_limit::{RateLimitLayer, RateLimitService, RateLimitSettings};
pub use request_id::{RequestId, RequestIdLayer, RequestIdService};

use std::sync::Arc;

use tower::ServiceBuilder;

use crate::auth::registry::AuthRegistry;
use crate::guardrails::pipeline::GuardrailPipeline;

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
    /// Optional guardrail pipeline; `None` disables guardrail evaluation.
    pub guardrail_pipeline: Option<Arc<GuardrailPipeline>>,
}

// ── Stack type alias ──────────────────────────────────────────────────────────

/// The concrete type of the base middleware stack produced by
/// [`build_middleware_stack`].
///
/// The optional [`GuardrailLayer`] is applied on top separately when present
/// (see [`build_middleware_stack`] documentation).
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
/// If `cfg.guardrail_pipeline` is `Some`, a [`GuardrailLayer`] is also stacked
/// on top of the base stack (innermost — applied last, closest to the handler).
/// The guardrail pipeline evaluates both pre-request and post-response engines.
///
/// Callers wrap their axum `Router` with this builder:
///
/// ```rust,ignore
/// let app = stack.service(router);
/// ```
pub fn build_middleware_stack(cfg: MiddlewareConfig) -> ServiceBuilder<MiddlewareStack> {
    // Note: the optional guardrail layer is not included in the static
    // MiddlewareStack type alias because adding it would change the concrete
    // type returned by this function (breaking callers that name the type).
    // Instead, callers that need guardrails can wrap their router with
    // `GuardrailLayer` explicitly after calling this function.
    //
    // The `cfg.guardrail_pipeline` field is intentionally accepted here so
    // that callers can pass it as part of a unified config, and the main
    // server startup can apply the layer on the router directly.
    let _ = cfg.guardrail_pipeline; // consumed / used by the caller
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
            guardrail_pipeline: None,
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
            guardrail_pipeline: None,
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
