//! Tower middleware stack for switchboard-server.
//!
//! # Layers (innermost → outermost, i.e. applied in reverse order)
//!
//! 1. [`request_id::RequestIdLayer`] — generates / propagates
//!    `X-Switchboard-Request-Id` on every request.
//! 2. [`model_override::ModelOverrideLayer`] — resolves the model and provider
//!    for proxy requests; sets [`model_override::ResolvedModel`],
//!    [`model_override::ProviderName`], and [`model_override::SelectionReasonExt`]
//!    in extensions.
//! 3. [`auth_layer::AuthLayer`] — validates `Authorization` header via
//!    [`crate::auth::registry::AuthRegistry`]; injects [`crate::auth::validator::ValidatedClient`]
//!    into request extensions.  Returns 401 on failure.
//! 4. [`rate_limit::RateLimitLayer`] — token-bucket rate limiter per user.
//!    Returns 429 when limits are exceeded.
//! 5. [`auth_inject::AuthInjectLayer`] — selects a key from the matching
//!    [`crate::key_pool::KeyPool`] (identified by [`model_override::ProviderName`])
//!    and inserts [`auth_inject::SelectedKeyId`] into extensions.
//!
//! # Layer ordering rationale
//!
//! - `ModelOverrideLayer` runs before `AuthLayer` so the provider name is
//!   known for the entire request lifecycle (including auth errors).
//! - `AuthInjectLayer` runs after `AuthLayer` so `ValidatedClient` is available
//!   for user-aware key-selection strategies (e.g. sticky selectors).
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

pub mod auth_inject;
pub mod auth_layer;
pub mod model_override;
pub mod rate_limit;
pub mod request_id;

pub use auth_inject::{AuthInjectLayer, AuthInjectService, SelectedKeyId};
pub use auth_layer::{AuthLayer, AuthService};
pub use model_override::{
    ModelOverrideLayer, ModelOverrideService, ProviderName, ResolvedModel, SelectionReasonExt,
};
pub use rate_limit::{RateLimitLayer, RateLimitService, RateLimitSettings};
pub use request_id::{RequestId, RequestIdLayer, RequestIdService};

use std::collections::HashMap;
use std::sync::Arc;

use tower::ServiceBuilder;

use crate::auth::registry::AuthRegistry;
use crate::config::provider::ProvidersConfig;
use crate::key_pool::KeyPool;
use crate::providers::ProviderRegistry;
use crate::routing::ModelSelector;

// ── Middleware configuration ───────────────────────────────────────────────────

/// Configuration for [`build_middleware_stack`].
pub struct MiddlewareConfig {
    /// Auth registry used by the auth layer.
    pub auth_registry: Arc<AuthRegistry>,
    /// Default requests-per-minute for users without an override.
    pub default_rpm: u32,
    /// Default tokens-per-minute for users without an override.
    pub default_tpm: u32,
    /// Per-user rate limit overrides.
    pub rate_limit_overrides: Vec<(String, RateLimitSettings)>,
    /// Key pools keyed by provider name — used by [`AuthInjectLayer`].
    pub key_pools: Arc<HashMap<String, Arc<KeyPool>>>,
    /// Model selector that applies the configured selection policy.
    pub model_selector: Arc<ModelSelector>,
    /// Registry of upstream providers — used by [`ModelOverrideLayer`].
    pub provider_registry: Arc<ProviderRegistry>,
    /// Providers configuration — used by [`ModelOverrideLayer`] to resolve
    /// which provider serves each model.
    pub providers_config: Arc<ProvidersConfig>,
}

impl std::fmt::Debug for MiddlewareConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiddlewareConfig")
            .field("default_rpm", &self.default_rpm)
            .field("default_tpm", &self.default_tpm)
            .field(
                "rate_limit_overrides_count",
                &self.rate_limit_overrides.len(),
            )
            .field("key_pool_count", &self.key_pools.len())
            .field("provider_registry", &self.provider_registry)
            .finish_non_exhaustive()
    }
}

// ── Stack type alias ──────────────────────────────────────────────────────────

/// The concrete type of the full middleware stack produced by
/// [`build_middleware_stack`].
///
/// Layer order (outermost → innermost, i.e. request traversal order):
/// `RequestIdLayer → ModelOverrideLayer → AuthLayer → RateLimitLayer → AuthInjectLayer`
pub type MiddlewareStack = tower::layer::util::Stack<
    AuthInjectLayer,
    tower::layer::util::Stack<
        RateLimitLayer,
        tower::layer::util::Stack<
            AuthLayer,
            tower::layer::util::Stack<
                ModelOverrideLayer,
                tower::layer::util::Stack<RequestIdLayer, tower::layer::util::Identity>,
            >,
        >,
    >,
>;

// ── Stack builder ─────────────────────────────────────────────────────────────

/// Build the standard switchboard-server Tower middleware stack.
///
/// Returns a [`ServiceBuilder`] with the following layers applied (outermost
/// first, i.e. the order in which a request passes through):
///
/// 1. `RequestIdLayer` — assign / propagate request ID
/// 2. `ModelOverrideLayer` — resolve model and provider
/// 3. `AuthLayer` — validate client credentials
/// 4. `RateLimitLayer` — enforce per-user rate limits
/// 5. `AuthInjectLayer` — select upstream key
///
/// Callers wrap their axum `Router` with this builder:
///
/// ```rust,ignore
/// let app = stack.service(router);
/// ```
pub fn build_middleware_stack(cfg: MiddlewareConfig) -> ServiceBuilder<MiddlewareStack> {
    ServiceBuilder::new()
        .layer(RequestIdLayer)
        .layer(ModelOverrideLayer::new(
            cfg.model_selector,
            cfg.provider_registry,
            cfg.providers_config,
        ))
        .layer(AuthLayer::new(cfg.auth_registry))
        .layer(RateLimitLayer::new(
            cfg.default_rpm,
            cfg.default_tpm,
            cfg.rate_limit_overrides,
        ))
        .layer(AuthInjectLayer::new(cfg.key_pools))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::auth::registry::AuthRegistry;
    use crate::config::model_selection::ModelSelectionConfig;
    use crate::providers::ProviderRegistry;
    use crate::routing::ModelSelector;

    fn make_config() -> MiddlewareConfig {
        MiddlewareConfig {
            auth_registry: Arc::new(AuthRegistry::new(vec![])),
            default_rpm: 60,
            default_tpm: 100_000,
            rate_limit_overrides: vec![],
            key_pools: Arc::new(HashMap::new()),
            model_selector: Arc::new(ModelSelector::new(ModelSelectionConfig::default())),
            provider_registry: Arc::new(ProviderRegistry::new()),
            providers_config: Arc::new(HashMap::new()),
        }
    }

    #[test]
    fn test_middleware_config_debug() {
        let cfg = make_config();
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
            key_pools: Arc::new(HashMap::new()),
            model_selector: Arc::new(ModelSelector::new(ModelSelectionConfig::default())),
            provider_registry: Arc::new(ProviderRegistry::new()),
            providers_config: Arc::new(HashMap::new()),
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

    #[test]
    fn test_middleware_config_with_overrides() {
        let cfg = MiddlewareConfig {
            auth_registry: Arc::new(AuthRegistry::new(vec![])),
            default_rpm: 10,
            default_tpm: 10_000,
            rate_limit_overrides: vec![(
                "vip".to_owned(),
                RateLimitSettings {
                    rpm: 1_000,
                    tpm: 5_000_000,
                },
            )],
            key_pools: Arc::new(HashMap::new()),
            model_selector: Arc::new(ModelSelector::new(ModelSelectionConfig::default())),
            provider_registry: Arc::new(ProviderRegistry::new()),
            providers_config: Arc::new(HashMap::new()),
        };
        assert_eq!(cfg.default_rpm, 10);
        assert_eq!(cfg.rate_limit_overrides.len(), 1);
    }
}
