//! switchboard-server library crate.
//!
//! Re-exports all public modules so they can be referenced from integration
//! tests without duplicating module declarations.

pub mod admin;
pub mod auth;
pub mod config;
pub mod error;
pub mod guardrails;
pub mod identity;
pub mod key_pool;
pub mod middleware;
#[allow(dead_code, unused_imports)]
pub mod observability;
pub mod providers;
pub mod proxy;
pub mod routing;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use axum::routing::{get, post};
use http::HeaderValue;

use crate::admin::auth::AdminAuthState;
use crate::admin::{AdminState, serve_admin, serve_admin_on_listener};
use crate::auth::UpstreamCredentials;
use crate::auth::registry::AuthRegistryBuilder;
use crate::auth::static_key::StaticKeyValidator;
use crate::config::HotConfig;
use crate::guardrails::pipeline::GuardrailPipeline;
use crate::key_pool::{
    AwsStsProvider, KeyPool, KeyProvider, KeySelector, LeastLoadedSelector, PooledKey,
    RoundRobinSelector, WeightedRandomSelector,
};
use crate::middleware::{
    GuardrailLayer, MiddlewareConfig, RateLimitLayer, RateLimitSettings, build_middleware_stack,
};
use crate::observability::UsageTracker;
use crate::providers::{
    AnthropicProvider, BedrockProvider, OllamaProvider, OpenAiProvider, ProviderRegistry,
    VertexProvider,
};
use crate::proxy::handler::{AppState, anthropic_messages, chat_completions, health, list_models};
use crate::routing::selector::ModelSelector;

/// Build an [`AuthRegistry`] from the server configuration.
///
/// Iterates `config.auth.validators` and registers each supported validator
/// type.  Unknown types are logged as warnings and skipped.
pub fn build_auth_registry(
    config: &crate::config::ServerConfig,
) -> crate::auth::registry::AuthRegistry {
    let mut builder = AuthRegistryBuilder::default();

    for (name, entry) in &config.auth.validators {
        match entry.validator_type.as_str() {
            "static_keys" => {
                builder = builder.add(StaticKeyValidator::new(name, entry.keys.clone()));
                tracing::info!(validator = name, "registered static_keys auth validator");
            }
            "jwt" => {
                tracing::warn!(
                    validator = name,
                    "JWT validator requires runtime JWKS fetch — skipping for now"
                );
            }
            other => {
                tracing::warn!(
                    validator = name,
                    validator_type = other,
                    "unsupported auth validator type, skipping"
                );
            }
        }
    }

    builder.build()
}

/// Build the [`ProviderRegistry`] and key pools from the server config.
pub async fn build_providers(
    config: &crate::config::ServerConfig,
) -> (
    crate::providers::ProviderRegistry,
    std::collections::HashMap<String, std::sync::Arc<crate::key_pool::KeyPool>>,
) {
    let mut registry = ProviderRegistry::new();
    let mut pools: HashMap<String, Arc<KeyPool>> = HashMap::new();

    for (name, provider_cfg) in &config.providers {
        // Build the provider implementation based on api_format.
        match provider_cfg.api_format.as_str() {
            "anthropic" => match AnthropicProvider::new(name, provider_cfg) {
                Ok(p) => {
                    registry.register(name, Arc::new(p));
                    tracing::info!(provider = name, "registered AnthropicProvider");
                }
                Err(e) => {
                    tracing::error!(provider = name, error = %e, "failed to build AnthropicProvider");
                }
            },
            "openai" => match OpenAiProvider::new(name, provider_cfg) {
                Ok(p) => {
                    registry.register(name, Arc::new(p));
                    tracing::info!(provider = name, "registered OpenAiProvider");
                }
                Err(e) => {
                    tracing::error!(provider = name, error = %e, "failed to build OpenAiProvider");
                }
            },
            "bedrock" => match BedrockProvider::new(name, provider_cfg) {
                Ok(p) => {
                    registry.register(name, Arc::new(p));
                    tracing::info!(provider = name, "registered BedrockProvider");
                }
                Err(e) => {
                    tracing::error!(provider = name, error = %e, "failed to build BedrockProvider");
                }
            },
            "vertex" => match VertexProvider::new(name, provider_cfg) {
                Ok(p) => {
                    registry.register(name, Arc::new(p));
                    tracing::info!(provider = name, "registered VertexProvider");
                }
                Err(e) => {
                    tracing::error!(provider = name, error = %e, "failed to build VertexProvider");
                }
            },
            "ollama" => match OllamaProvider::new(name, provider_cfg) {
                Ok(p) => {
                    registry.register(name, Arc::new(p));
                    tracing::info!(provider = name, "registered OllamaProvider");
                }
                Err(e) => {
                    tracing::error!(provider = name, error = %e, "failed to build OllamaProvider");
                }
            },
            other => {
                tracing::warn!(
                    provider = name,
                    api_format = other,
                    "unsupported api_format, skipping"
                );
                continue;
            }
        }

        // Build the key pool for this provider.
        let mut keys: Vec<PooledKey> = Vec::new();
        for entry in &provider_cfg.key_pool.keys {
            match entry.key_type.as_str() {
                "static" => {
                    let api_key = entry.api_key.as_deref().unwrap_or("");
                    let (header_name, header_value_str) = match provider_cfg.api_format.as_str() {
                        "anthropic" => (
                            http::HeaderName::from_static("x-api-key"),
                            api_key.to_string(),
                        ),
                        "bedrock" => (
                            http::HeaderName::from_static("x-switchboard-bedrock-creds"),
                            api_key.to_string(),
                        ),
                        _ => (
                            http::HeaderName::from_static("authorization"),
                            format!("Bearer {api_key}"),
                        ),
                    };
                    let header_value = HeaderValue::from_str(&header_value_str)
                        .unwrap_or_else(|_| HeaderValue::from_static("invalid"));
                    let creds = UpstreamCredentials {
                        header_name,
                        header_value,
                        expires_at: None,
                    };
                    keys.push(PooledKey::new_static(&entry.id, creds, entry.weight));
                }
                "aws_sts" => {
                    let role_arn = match entry.role_arn.as_deref() {
                        Some(r) => r,
                        None => {
                            tracing::error!(
                                key_id = %entry.id,
                                "aws_sts key entry missing role_arn, skipping"
                            );
                            continue;
                        }
                    };
                    let region = entry
                        .region
                        .as_deref()
                        .or(provider_cfg.region.as_deref())
                        .unwrap_or("us-east-1");
                    let provider = AwsStsProvider::new(role_arn, region);
                    match provider.fetch().await {
                        Ok(creds) => {
                            tracing::info!(
                                key_id = %entry.id,
                                role_arn = role_arn,
                                "fetched initial STS credentials for key"
                            );
                            keys.push(PooledKey {
                                id: entry.id.clone(),
                                credentials: creds,
                                weight: entry.weight,
                                source: crate::key_pool::KeySource::AwsSts {
                                    role_arn: role_arn.to_string(),
                                },
                                health: crate::key_pool::KeyHealth::default(),
                            });
                        }
                        Err(e) => {
                            tracing::error!(
                                key_id = %entry.id,
                                role_arn = role_arn,
                                error = %e,
                                "failed to fetch initial STS credentials, skipping key"
                            );
                        }
                    }
                }
                "vault" => {
                    let vault_addr = std::env::var("VAULT_ADDR")
                        .unwrap_or_else(|_| "http://127.0.0.1:8200".to_string());
                    let vault_token = std::env::var("VAULT_TOKEN").unwrap_or_default();
                    let path = entry.vault_path.as_deref().unwrap_or("");

                    let provider =
                        crate::key_pool::VaultProvider::new(path, &vault_addr, &vault_token);
                    match provider.fetch().await {
                        Ok(creds) => {
                            let expires_at = creds.expires_at;
                            if let Some(exp) = expires_at {
                                tracing::info!(
                                    key_id = %entry.id,
                                    vault_path = path,
                                    expires_in_secs = exp.duration_since(std::time::Instant::now()).as_secs(),
                                    "fetched initial Vault credentials with TTL"
                                );
                            } else {
                                tracing::info!(
                                    key_id = %entry.id,
                                    vault_path = path,
                                    "fetched initial Vault credentials (no TTL)"
                                );
                            }
                            keys.push(PooledKey {
                                id: entry.id.clone(),
                                credentials: creds,
                                weight: entry.weight,
                                source: crate::key_pool::KeySource::Vault {
                                    path: path.to_string(),
                                },
                                health: crate::key_pool::KeyHealth::default(),
                            });
                        }
                        Err(e) => {
                            tracing::error!(
                                key_id = %entry.id,
                                vault_path = path,
                                error = %e,
                                "failed to fetch initial Vault credentials, skipping key"
                            );
                        }
                    }
                }
                other => {
                    tracing::warn!(
                        key_id = %entry.id,
                        key_type = other,
                        "unsupported key type, skipping"
                    );
                }
            }
        }

        let selector: Box<dyn KeySelector> = match provider_cfg.key_pool.selector.as_str() {
            "round_robin" => Box::new(RoundRobinSelector::new()),
            "least_loaded" => Box::new(LeastLoadedSelector),
            _ => Box::new(WeightedRandomSelector),
        };

        let pool = KeyPool::new(keys, selector);
        pools.insert(name.clone(), Arc::new(pool));
        tracing::info!(provider = name, "built key pool");
    }

    (registry, pools)
}

/// Start the full switchboard server.
///
/// Accepts a pre-bound [`tokio::net::TcpListener`] for the proxy and an
/// optional pre-bound listener for the admin server.  This signature makes it
/// easy for integration tests to bind to port `0`, discover the OS-assigned
/// port, and then hand the listener here — avoiding TOCTOU races.
///
/// # Parameters
///
/// - `config` — Fully-loaded server configuration.
/// - `config_path` — Filesystem path used to construct the [`HotConfig`] (for
///   live-reload support).  May be an empty string in tests.
/// - `listener` — Pre-bound TCP listener for the proxy endpoint.
/// - `admin_listener` — Pre-bound TCP listener for the admin endpoint.  If
///   `None` and `config.admin.enabled` is `true`, the address from config is
///   bound internally.
/// - `shutdown` — Future that resolves when the server should begin graceful
///   shutdown.
pub async fn run_server(
    config: crate::config::ServerConfig,
    config_path: &str,
    listener: tokio::net::TcpListener,
    admin_listener: Option<tokio::net::TcpListener>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<()> {
    // Extract fields needed after config is moved into AppState.
    let shutdown_timeout = crate::config::duration::parse(&config.server.graceful_shutdown_timeout)
        .unwrap_or(std::time::Duration::from_secs(30));

    // Extract sub-configs needed for middleware before server_config is
    // moved into AppState.
    let auth_registry = build_auth_registry(&config);
    let model_selector = Arc::new(ModelSelector::new(config.model_selection.clone()));
    let rate_limit_overrides: Vec<(String, RateLimitSettings)> = config
        .rate_limit
        .overrides
        .iter()
        .map(|(user, ov)| {
            (
                user.clone(),
                RateLimitSettings {
                    rpm: ov.rpm.unwrap_or(config.rate_limit.default_rpm),
                    tpm: ov.tpm.unwrap_or(config.rate_limit.default_tpm),
                },
            )
        })
        .collect();
    let default_rpm = config.rate_limit.default_rpm;
    let default_tpm = config.rate_limit.default_tpm;
    let providers_config = Arc::new(config.providers.clone());
    let guardrails_config = config.guardrails.clone();

    // Build provider registry and key pools from config.
    let (provider_registry, key_pools) = build_providers(&config).await;

    // Create a shared usage tracker — the same Arc is held by both AppState
    // (written by proxy handlers) and AdminState (read by admin API).
    let usage = Arc::new(UsageTracker::new());

    // Pre-build the rate limit layer and handle so the handle can be shared
    // with the admin server.  Both reference the same underlying DashMap, so
    // changes via the handle are immediately visible to in-flight requests.
    let (rate_limit_layer, rate_limit_handle) =
        RateLimitLayer::new(default_rpm, default_tpm, rate_limit_overrides);

    let app_state = Arc::new(AppState {
        config: Arc::new(config),
        providers: Arc::new(provider_registry),
        key_pools: Arc::new(key_pools),
        usage: Arc::clone(&usage),
    });

    // Spawn the admin server if enabled.
    if app_state.config.admin.enabled {
        let mut admin_auth_state_inner = AdminAuthState::new(app_state.config.admin.clone());
        if app_state.config.admin.auth == "jwt" {
            if let Err(e) = admin_auth_state_inner.init_jwt().await {
                tracing::warn!(error = %e, "admin JWT init failed, continuing without JWT auth");
            }
        }
        let admin_auth_state = Arc::new(admin_auth_state_inner);

        // Build admin-writable key pools (RwLock-wrapped for admin mutations).
        // One empty pool per provider is created; the admin API populates them
        // at runtime. Proxy reads use the original Arc<KeyPool> independently.
        let admin_pools: std::collections::HashMap<String, Arc<std::sync::RwLock<KeyPool>>> =
            app_state
                .key_pools
                .keys()
                .map(|k| {
                    let pool = KeyPool::new(vec![], Box::new(WeightedRandomSelector));
                    (k.clone(), Arc::new(std::sync::RwLock::new(pool)))
                })
                .collect();

        let admin_state = Arc::new(AdminState::new(
            Arc::new(HotConfig::new(
                (*app_state.config).clone(),
                config_path.to_string(),
            )),
            Arc::new(admin_pools),
            admin_auth_state,
            Arc::clone(&usage),
            rate_limit_handle.clone(),
        ));

        let admin_listen = app_state.config.admin.listen.clone();
        tokio::spawn(async move {
            let result = if let Some(l) = admin_listener {
                serve_admin_on_listener(admin_state, l).await
            } else {
                serve_admin(admin_state, &admin_listen).await
            };
            if let Err(e) = result {
                tracing::error!(error = %e, "admin server error");
            }
        });
        tracing::info!(addr = %app_state.config.admin.listen, "admin server spawned");
    }

    // Build guardrail pipeline from config (if enabled).
    let guardrail_pipeline = if guardrails_config.enabled {
        match GuardrailPipeline::from_config(&guardrails_config) {
            Ok(pipeline) => {
                tracing::info!("guardrail pipeline enabled");
                Some(Arc::new(pipeline))
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to build guardrail pipeline, continuing without guardrails");
                None
            }
        }
    } else {
        tracing::debug!("guardrails disabled");
        None
    };

    let middleware_cfg = MiddlewareConfig {
        auth_registry: Arc::new(auth_registry),
        default_rpm,
        default_tpm,
        rate_limit_overrides: vec![],
        key_pools: Arc::clone(&app_state.key_pools),
        model_selector,
        provider_registry: Arc::clone(&app_state.providers),
        providers_config,
        guardrail_pipeline: guardrail_pipeline.clone(),
        // Supply the pre-built layer so the admin server's handle shares the
        // same underlying DashMap.
        rate_limit_layer: Some((rate_limit_layer, rate_limit_handle.clone())),
    };

    let (stack, _stack_handle) = build_middleware_stack(middleware_cfg);

    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/api/v1/messages", post(anthropic_messages))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(app_state);

    // Apply guardrail layer closest to the handler (before middleware stack).
    if let Some(pipeline) = guardrail_pipeline {
        router = router.layer(GuardrailLayer::new(pipeline));
        tracing::info!("guardrail layer applied to router");
    }

    // Apply the middleware stack (outermost layers).
    let router = router.layer(stack);
    tracing::info!("middleware stack applied");

    let listen_addr = listener.local_addr()?;
    tracing::info!(addr = %listen_addr, "listening");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await?;

    let _ = shutdown_timeout; // retain for future use in drain logic
    tracing::info!("server shut down cleanly");
    Ok(())
}
