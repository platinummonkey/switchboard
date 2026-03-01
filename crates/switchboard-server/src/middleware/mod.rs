//! Tower middleware stack for switchboard-server.
//!
//! # Layers (outermost → innermost, i.e. request traversal order)
//!
//! 1. [`request_id::RequestIdLayer`] — generates / propagates
//!    `X-Switchboard-Request-Id` on every request.
//! 2. [`auth_layer::AuthLayer`] — validates `Authorization` header via
//!    [`crate::auth::registry::AuthRegistry`]; injects [`crate::auth::validator::ValidatedClient`]
//!    into request extensions.  Returns 401 on failure.
//! 3. [`model_override::ModelOverrideLayer`] — resolves the model and provider
//!    for proxy requests; sets [`model_override::ResolvedModel`],
//!    [`model_override::ProviderName`], and [`model_override::SelectionReasonExt`]
//!    in extensions.  Reads `ValidatedClient.user_id` (injected by the auth
//!    layer) so that per-user model overrides fire for api-key-mapped users
//!    even when no explicit `x-switchboard-user` header is present.
//! 4. [`rate_limit::RateLimitLayer`] — token-bucket rate limiter per user.
//!    Returns 429 when limits are exceeded.
//! 5. [`auth_inject::AuthInjectLayer`] — selects a key from the matching
//!    [`crate::key_pool::KeyPool`] (identified by [`model_override::ProviderName`])
//!    and inserts [`auth_inject::SelectedKeyId`] into extensions.
//! 6. [`guardrail::GuardrailLayer`] — (optional) evaluates guardrail engines
//!    before forwarding requests and after receiving responses.
//!
//! # Layer ordering rationale
//!
//! - `AuthLayer` runs before `ModelOverrideLayer` so that `ValidatedClient`
//!   (with the resolved `user_id` from api-key mappings) is available when the
//!   model selector evaluates per-user overrides.  This is the key enabler for
//!   api-key-authenticated users to receive per-user model policies without
//!   needing an explicit `x-switchboard-user` header.
//! - `ModelOverrideLayer` runs after `AuthLayer` so it can read the
//!   `ValidatedClient` extension and fall back to `ValidatedClient.user_id`
//!   when the `x-switchboard-user` header is absent.
//! - `AuthInjectLayer` runs after `AuthLayer` so `ValidatedClient` is available
//!   for user-aware key-selection strategies (e.g. sticky selectors).
//! - `GuardrailLayer` wraps closest to the handler so it sees the final
//!   resolved request body and the actual upstream response.
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
//! let (stack, handle) = build_middleware_stack(MiddlewareConfig {
//!     auth_registry: registry,
//!     default_rpm: 60,
//!     default_tpm: 100_000,
//!     rate_limit_overrides: vec![],
//!     key_pools: Arc::new(std::collections::HashMap::new()),
//!     model_selector: Arc::new(ModelSelector::new(ModelSelectionConfig::default())),
//!     provider_registry: Arc::new(ProviderRegistry::new()),
//!     providers_config: Arc::new(std::collections::HashMap::new()),
//!     guardrail_pipeline: None,
//!     rate_limit_layer: None,
//! });
//! ```

pub mod auth_inject;
pub mod auth_layer;
pub mod guardrail;
pub mod model_override;
pub mod rate_limit;
pub mod request_id;

pub use auth_inject::{AuthInjectLayer, AuthInjectService, SelectedKeyId};
pub use auth_layer::{AuthLayer, AuthService};
pub use guardrail::GuardrailLayer;
pub use model_override::{
    ModelOverrideLayer, ModelOverrideService, ProviderName, ResolvedModel, SelectionReasonExt,
};
pub use rate_limit::{RateLimitHandle, RateLimitLayer, RateLimitService, RateLimitSettings};
pub use request_id::{RequestId, RequestIdLayer, RequestIdService};

use std::collections::HashMap;
use std::sync::Arc;

use tower::ServiceBuilder;

use crate::auth::registry::AuthRegistry;
use crate::config::provider::ProvidersConfig;
use crate::guardrails::pipeline::GuardrailPipeline;
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
    /// Optional guardrail pipeline; `None` disables guardrail evaluation.
    pub guardrail_pipeline: Option<Arc<GuardrailPipeline>>,
    /// Optional pre-built rate limit layer.  When `Some`, the layer and its
    /// handle are used directly (allowing the handle to be shared with the
    /// admin server).  When `None`, a new layer is created from the
    /// `default_rpm`/`default_tpm`/`rate_limit_overrides` fields.
    pub rate_limit_layer: Option<(RateLimitLayer, RateLimitHandle)>,
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
            .field("guardrail_enabled", &self.guardrail_pipeline.is_some())
            .finish_non_exhaustive()
    }
}

// ── Stack type alias ──────────────────────────────────────────────────────────

/// The concrete type of the base middleware stack produced by
/// [`build_middleware_stack`].
///
/// Layer order (outermost → innermost, i.e. request traversal order):
/// `RequestIdLayer → AuthLayer → ModelOverrideLayer → RateLimitLayer → AuthInjectLayer`
///
/// `AuthLayer` runs before `ModelOverrideLayer` so that `ValidatedClient`
/// (carrying `user_id` resolved from api-key mappings) is available when the
/// model selector evaluates per-user overrides.
///
/// The optional [`GuardrailLayer`] is applied on the router directly when
/// `cfg.guardrail_pipeline` is `Some` — it is not included in this type alias
/// to avoid changing the concrete return type.
pub type MiddlewareStack = tower::layer::util::Stack<
    AuthInjectLayer,
    tower::layer::util::Stack<
        RateLimitLayer,
        tower::layer::util::Stack<
            ModelOverrideLayer,
            tower::layer::util::Stack<
                AuthLayer,
                tower::layer::util::Stack<RequestIdLayer, tower::layer::util::Identity>,
            >,
        >,
    >,
>;

// ── Stack builder ─────────────────────────────────────────────────────────────

/// Build the standard switchboard-server Tower middleware stack.
///
/// Returns `(stack, handle)` where:
/// - `stack` is a [`ServiceBuilder`] with the following layers applied
///   (outermost first, i.e. the order in which a request passes through):
///   1. `RequestIdLayer` — assign / propagate request ID
///   2. `AuthLayer` — validate client credentials; injects [`crate::auth::validator::ValidatedClient`]
///      (with `user_id` resolved from api-key mappings) into extensions.
///   3. `ModelOverrideLayer` — resolve model and provider; reads `ValidatedClient.user_id`
///      as a fallback when the `x-switchboard-user` header is absent, enabling
///      per-user model overrides for api-key-authenticated users.
///   4. `RateLimitLayer` — enforce per-user rate limits
///   5. `AuthInjectLayer` — select upstream key
/// - `handle` is a [`RateLimitHandle`] that allows live updates to rate-limit
///   overrides without restarting.  Pass it to [`crate::admin::AdminState`] so
///   the admin API can mutate it at runtime.
///
/// If `cfg.guardrail_pipeline` is `Some`, the caller should wrap the axum
/// `Router` with [`GuardrailLayer`] directly (see `main.rs`).
pub fn build_middleware_stack(
    cfg: MiddlewareConfig,
) -> (ServiceBuilder<MiddlewareStack>, RateLimitHandle) {
    // guardrail_pipeline is consumed by the caller — it is available in cfg so
    // the main server can extract it and apply GuardrailLayer on the router.
    let _ = cfg.guardrail_pipeline;

    // Use a pre-built layer+handle when provided (allows sharing the handle
    // with the admin server); otherwise construct from the config fields.
    let (rate_limit_layer, handle) = cfg.rate_limit_layer.unwrap_or_else(|| {
        RateLimitLayer::new(cfg.default_rpm, cfg.default_tpm, cfg.rate_limit_overrides)
    });

    // AuthLayer runs before ModelOverrideLayer so that ValidatedClient
    // (carrying user_id resolved from api-key mappings by StaticKeyValidator)
    // is available in extensions when the model selector evaluates per-user
    // overrides.  ModelOverrideLayer reads ValidatedClient.user_id as a
    // fallback when no explicit x-switchboard-user header is present.
    let stack = ServiceBuilder::new()
        .layer(RequestIdLayer)
        .layer(AuthLayer::new(cfg.auth_registry))
        .layer(ModelOverrideLayer::new(
            cfg.model_selector,
            cfg.provider_registry,
            cfg.providers_config,
        ))
        .layer(rate_limit_layer)
        .layer(AuthInjectLayer::new(cfg.key_pools));
    (stack, handle)
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
            guardrail_pipeline: None,
            rate_limit_layer: None,
        }
    }

    #[test]
    fn test_middleware_config_debug() {
        let cfg = make_config();
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
            guardrail_pipeline: None,
            rate_limit_layer: None,
        };
        let (_stack, _handle) = build_middleware_stack(cfg);
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
            guardrail_pipeline: None,
            rate_limit_layer: None,
        };
        assert_eq!(cfg.default_rpm, 10);
        assert_eq!(cfg.rate_limit_overrides.len(), 1);
    }

    #[test]
    fn test_guardrail_pipeline_none_by_default() {
        let cfg = make_config();
        assert!(cfg.guardrail_pipeline.is_none());
    }
}
