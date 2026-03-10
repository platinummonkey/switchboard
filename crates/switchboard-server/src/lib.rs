//! switchboard-server library crate.
//!
//! Re-exports all public modules so they can be referenced from integration
//! tests without duplicating module declarations.

pub mod admin;
pub mod auth;
pub mod config;
pub mod db;
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
pub mod tls;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::db::DbPool;

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
    RoundRobinSelector, StickySelector, WeightedRandomSelector,
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
use crate::routing::ProviderHealthChecker;
use crate::routing::selector::ModelSelector;

// ── Key provider AuthProvider adapters for spawn_refresh_task ─────────────────

struct AwsStsAuthAdapter(AwsStsProvider);

#[async_trait::async_trait]
impl crate::auth::AuthProvider for AwsStsAuthAdapter {
    fn name(&self) -> &str {
        "aws_sts"
    }
    async fn get_credentials(
        &self,
    ) -> Result<crate::auth::UpstreamCredentials, crate::auth::UpstreamAuthError> {
        self.0
            .fetch()
            .await
            .map_err(|e| crate::auth::UpstreamAuthError::FetchFailed(e.to_string()))
    }
    async fn refresh(
        &self,
    ) -> Result<crate::auth::UpstreamCredentials, crate::auth::UpstreamAuthError> {
        self.get_credentials().await
    }
    fn is_valid(&self) -> bool {
        true
    }
}

struct VaultAuthAdapter(crate::key_pool::VaultProvider);

#[async_trait::async_trait]
impl crate::auth::AuthProvider for VaultAuthAdapter {
    fn name(&self) -> &str {
        "vault"
    }
    async fn get_credentials(
        &self,
    ) -> Result<crate::auth::UpstreamCredentials, crate::auth::UpstreamAuthError> {
        self.0
            .fetch()
            .await
            .map_err(|e| crate::auth::UpstreamAuthError::FetchFailed(e.to_string()))
    }
    async fn refresh(
        &self,
    ) -> Result<crate::auth::UpstreamCredentials, crate::auth::UpstreamAuthError> {
        self.get_credentials().await
    }
    fn is_valid(&self) -> bool {
        true
    }
}

/// Build an [`AuthRegistry`] from the server configuration.
///
/// Iterates `config.auth.validators` and registers each supported validator
/// type.  Unknown types are logged as warnings and skipped.
///
/// For `static_keys` validators, api-key → user-identity mappings from
/// `config.identity.api_key_mappings` are forwarded to the validator so that
/// requests authenticated via a static key automatically resolve a `user_id`
/// on the returned [`crate::auth::validator::ValidatedClient`].  This allows
/// per-user model overrides (`ModelSelectionConfig.overrides`) and other
/// identity-aware policies to fire for api-key-mapped users without requiring
/// an explicit `x-switchboard-user` header.
pub fn build_auth_registry(
    config: &crate::config::ServerConfig,
) -> crate::auth::registry::AuthRegistry {
    let mut builder = AuthRegistryBuilder::default();

    // Build the user_map once from identity.api_key_mappings so it can be
    // shared across all static_keys validators (most configs have only one).
    let user_map: std::collections::HashMap<String, (String, Option<String>)> = config
        .identity
        .api_key_mappings
        .iter()
        .map(|m| (m.api_key.clone(), (m.user_id.clone(), m.team.clone())))
        .collect();

    for (name, entry) in &config.auth.validators {
        match entry.validator_type.as_str() {
            "static_keys" => {
                builder = builder.add(StaticKeyValidator::new_with_mappings(
                    name,
                    entry.keys.clone(),
                    user_map.clone(),
                ));
                tracing::info!(
                    validator = name,
                    mapped_identities = user_map.len(),
                    "registered static_keys auth validator"
                );
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
        let mut keys: Vec<Arc<RwLock<PooledKey>>> = Vec::new();
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
                    keys.push(Arc::new(RwLock::new(PooledKey::new_static(
                        &entry.id,
                        creds,
                        entry.weight,
                    ))));
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
                            let key_arc = Arc::new(RwLock::new(PooledKey {
                                id: entry.id.clone(),
                                credentials: creds,
                                weight: entry.weight,
                                source: crate::key_pool::KeySource::AwsSts {
                                    role_arn: role_arn.to_string(),
                                },
                                health: crate::key_pool::KeyHealth::default(),
                            }));
                            // Spawn proactive refresh for STS key.
                            {
                                let auth = Arc::new(AwsStsAuthAdapter(provider.clone()));
                                crate::key_pool::spawn_refresh_task(
                                    Arc::clone(&key_arc),
                                    auth,
                                    std::time::Duration::from_secs(30),
                                );
                                tracing::info!(key_id = %entry.id, "spawned STS refresh task");
                            }
                            keys.push(key_arc);
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
                            let key_arc = Arc::new(RwLock::new(PooledKey {
                                id: entry.id.clone(),
                                credentials: creds,
                                weight: entry.weight,
                                source: crate::key_pool::KeySource::Vault {
                                    path: path.to_string(),
                                },
                                health: crate::key_pool::KeyHealth::default(),
                            }));
                            // Spawn proactive refresh for Vault key.
                            {
                                let auth = Arc::new(VaultAuthAdapter(provider.clone()));
                                crate::key_pool::spawn_refresh_task(
                                    Arc::clone(&key_arc),
                                    auth,
                                    std::time::Duration::from_secs(30),
                                );
                                tracing::info!(key_id = %entry.id, "spawned Vault refresh task");
                            }
                            keys.push(key_arc);
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

        // For the least_loaded selector we need to share the in-flight tracker
        // between the selector (which reads it to pick the minimum) and the
        // KeyPool (which exposes it to AuthInjectService for write access).
        let (selector, tracker): (Box<dyn KeySelector>, _) =
            match provider_cfg.key_pool.selector.as_str() {
                "round_robin" => (Box::new(RoundRobinSelector::new()), None),
                "least_loaded" => {
                    let s = LeastLoadedSelector::new();
                    let t = s.in_flight_tracker();
                    (Box::new(s), Some(t))
                }
                "sticky" => (
                    Box::new(StickySelector::new(Box::new(RoundRobinSelector::new()))),
                    None,
                ),
                _ => (Box::new(WeightedRandomSelector), None),
            };

        let pool = KeyPool::new_with_tracker(keys, selector, tracker);
        pools.insert(name.clone(), Arc::new(pool));
        tracing::info!(provider = name, "built key pool");
    }

    (registry, pools)
}

/// Apply DB-persisted overrides onto `config` before the server starts.
///
/// - Rate limit overrides from `rate_limit_overrides` are merged into
///   `config.rate_limit.overrides`.
/// - Config snapshots for `model_selection`, `guardrails`, and `routing` from
///   `config_overrides` replace the corresponding config sections.
/// - Key pool entries from `key_pool_entries` update weights on existing keys
///   and reconstruct non-static (aws_sts/vault) keys that are new.  Static
///   keys that only exist in the DB (not TOML) are skipped with a warning.
async fn apply_db_overrides(
    pool: &DbPool,
    config: &mut crate::config::ServerConfig,
) -> Result<(), crate::error::ServerError> {
    use crate::config::rate_limit::RateLimitOverride;
    use crate::db::queries;

    // ── Rate limit overrides ────────────────────────────────────────────────
    let rl_rows = queries::list_rate_limit_overrides(pool.read()).await?;
    for row in rl_rows {
        config.rate_limit.overrides.insert(
            row.id.clone(),
            RateLimitOverride {
                rpm: Some(row.rpm as u32),
                tpm: Some(row.tpm as u32),
            },
        );
        tracing::debug!(id = %row.id, "loaded rate limit override from DB");
    }

    // ── Config section overrides ────────────────────────────────────────────
    for section in ["model_selection", "guardrails", "routing"] {
        let Some(row) = queries::get_config_override(pool.read(), section).await? else {
            continue;
        };
        match section {
            "model_selection" => match serde_json::from_value(row.config_json) {
                Ok(ms) => {
                    config.model_selection = ms;
                    tracing::info!("applied model_selection override from DB");
                }
                Err(e) => {
                    tracing::warn!(section, error = %e, "failed to deserialize config override, skipping");
                }
            },
            "guardrails" => match serde_json::from_value(row.config_json) {
                Ok(g) => {
                    config.guardrails = g;
                    tracing::info!("applied guardrails override from DB");
                }
                Err(e) => {
                    tracing::warn!(section, error = %e, "failed to deserialize config override, skipping");
                }
            },
            "routing" => match serde_json::from_value(row.config_json) {
                Ok(r) => {
                    config.routing = r;
                    tracing::info!("applied routing override from DB");
                }
                Err(e) => {
                    tracing::warn!(section, error = %e, "failed to deserialize config override, skipping");
                }
            },
            _ => {}
        }
    }

    // ── Key pool entries ────────────────────────────────────────────────────
    let key_rows = queries::list_key_pool_entries(pool.read()).await?;
    for row in key_rows {
        let Some(provider_cfg) = config.providers.get_mut(&row.provider_id) else {
            tracing::warn!(
                provider_id = %row.provider_id,
                key_id = %row.id,
                "key pool entry references unknown provider, skipping"
            );
            continue;
        };

        if let Some(entry) = provider_cfg
            .key_pool
            .keys
            .iter_mut()
            .find(|k| k.id == row.id)
        {
            // Update weight on an existing TOML key.
            entry.weight = row.weight;
            tracing::debug!(key_id = %row.id, weight = row.weight, "updated key weight from DB");
        } else if row.key_type != "static" {
            // Reconstruct non-static keys from source_config.
            let entry = crate::config::provider::KeyEntry {
                id: row.id.clone(),
                key_type: row.key_type.clone(),
                api_key: None,
                role_arn: row
                    .source_config
                    .get("role_arn")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned),
                region: row
                    .source_config
                    .get("region")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned),
                refresh_interval: None,
                vault_path: row
                    .source_config
                    .get("vault_path")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned),
                weight: row.weight,
            };
            provider_cfg.key_pool.keys.push(entry);
            tracing::info!(
                provider_id = %row.provider_id,
                key_id = %row.id,
                key_type = %row.key_type,
                "reconstructed dynamic key from DB"
            );
        } else {
            // Static key in DB but not TOML — skip (credential not stored in DB).
            tracing::warn!(
                provider_id = %row.provider_id,
                key_id = %row.id,
                "static key found in DB but not in TOML config, skipping"
            );
        }
    }

    Ok(())
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
    mut config: crate::config::ServerConfig,
    config_path: &str,
    listener: tokio::net::TcpListener,
    admin_listener: Option<tokio::net::TcpListener>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<()> {
    // Extract fields needed after config is moved into AppState.
    let shutdown_timeout = crate::config::duration::parse(&config.server.graceful_shutdown_timeout)
        .unwrap_or(std::time::Duration::from_secs(30));

    // Clone the listen config before config is moved into Arc<AppState>.
    // Needed to call build_tls_acceptor after AppState is constructed.
    let server_listen_config = config.server.clone();

    // ── DB init (optional) ─────────────────────────────────────────────────
    // Connect, migrate, and overlay DB-persisted overrides onto the in-memory
    // config before building the rest of the server state.  When
    // `database.enabled = false` this is a no-op and all existing tests pass
    // without any Postgres instance.
    let db_pool: Option<Arc<DbPool>> = if config.database.enabled {
        tracing::info!("database enabled, connecting...");
        let pool = DbPool::connect(&config.database).await?;
        if config.database.auto_migrate {
            pool.migrate().await?;
        }
        apply_db_overrides(&pool, &mut config).await?;
        tracing::info!("database overrides applied");
        Some(Arc::new(pool))
    } else {
        None
    };

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
    let (mut provider_registry, key_pools) = build_providers(&config).await;

    // Create the health checker early so it can be wired into both
    // ProviderRegistry (for resolve_provider filtering) and AppState (for
    // handler-level health checks). Must be created before registry is wrapped
    // in Arc.
    let health_checker = Arc::new(ProviderHealthChecker::new());
    provider_registry.set_health_checker(Arc::clone(&health_checker));

    // Create a shared usage tracker — the same Arc is held by both AppState
    // (written by proxy handlers) and AdminState (read by admin API).
    let usage = Arc::new(UsageTracker::new());

    // Pre-build the rate limit layer and handle so the handle can be shared
    // with the admin server.  Both reference the same underlying DashMap, so
    // changes via the handle are immediately visible to in-flight requests.
    let (rate_limit_layer, rate_limit_handle) =
        RateLimitLayer::new(default_rpm, default_tpm, rate_limit_overrides);

    // Create health checker (stored in AppState for fail-open access in handlers).
    let health_checker = Arc::new(ProviderHealthChecker::new());

    let app_state = Arc::new(AppState {
        config: Arc::new(config),
        providers: Arc::new(provider_registry),
        key_pools: Arc::new(key_pools),
        usage: Arc::clone(&usage),
        rate_limit_handle: Some(rate_limit_handle.clone()),
        health_checker: Arc::clone(&health_checker),
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
            db_pool.clone(),
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

    // Spawn background provider health checks.
    //
    // The health checker is the same Arc shared with ProviderRegistry and
    // AppState, so health results immediately influence both routing decisions
    // (resolve_provider filtering) and handler-level checks.
    {
        let health_interval = app_state
            .config
            .providers
            .values()
            .filter_map(|p| crate::config::duration::parse(&p.health_check_interval).ok())
            .min()
            .unwrap_or(std::time::Duration::from_secs(30));

        Arc::clone(&health_checker).spawn_health_checks(
            Arc::clone(&app_state.providers),
            Arc::clone(&app_state.key_pools),
            health_interval,
        );
        tracing::info!(
            interval_secs = health_interval.as_secs(),
            "provider health checker spawned"
        );
    }

    // Build guardrail pipeline from config (if enabled).
    // Use the async variant so that gRPC callout engines (which require an
    // async connection step) are handled correctly alongside builtin types.
    let guardrail_pipeline = if guardrails_config.enabled {
        match GuardrailPipeline::from_config_async(&guardrails_config).await {
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

    // Build the LLM proxy sub-router.  This is the only set of routes that
    // should be evaluated by the guardrail pipeline — the `/health` endpoint
    // must stay outside the guardrail layer so health checks are never blocked
    // by a guardrail returning a non-Pass verdict.
    let proxy_router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/api/v1/messages", post(anthropic_messages))
        .route("/v1/models", get(list_models));

    // Apply guardrail layer only to the proxy routes.
    let proxy_router: Router<Arc<AppState>> = if let Some(pipeline) = guardrail_pipeline {
        tracing::info!("guardrail layer applied to proxy routes");
        proxy_router.layer(GuardrailLayer::new(pipeline))
    } else {
        proxy_router
    };

    // Assemble the full router: proxy routes (with optional guardrails) +
    // health check (always unguarded). Apply shared AppState after merging so
    // both sub-routers receive the same state instance.
    let router = Router::new()
        .merge(proxy_router)
        .route("/health", get(health))
        .with_state(app_state);

    // Apply the middleware stack (outermost layers).
    let router = router.layer(stack);
    tracing::info!("middleware stack applied");

    // Try to build a TLS acceptor.  If TLS is not configured `tls_acceptor` is
    // `None` and we fall through to the plain-TCP path.
    let tls_acceptor = crate::tls::build_tls_acceptor(&server_listen_config).await?;

    if let Some(acceptor) = tls_acceptor {
        // TLS/mTLS path: custom accept loop using tokio-rustls + hyper-util.
        use hyper_util::rt::{TokioExecutor, TokioIo};
        use hyper_util::server::conn::auto::Builder as AutoConnBuilder;

        let auto_builder = AutoConnBuilder::new(TokioExecutor::new());
        let listen_addr = listener.local_addr()?;
        tracing::info!(addr = %listen_addr, "listening (TLS)");

        let shutdown = std::pin::pin!(shutdown);
        let mut shutdown = shutdown;

        loop {
            tokio::select! {
                biased;
                _ = &mut shutdown => {
                    tracing::info!(
                        drain_secs = shutdown_timeout.as_secs_f64(),
                        "TLS server: graceful shutdown, draining in-flight requests"
                    );
                    tokio::time::sleep(shutdown_timeout).await;
                    break;
                }
                result = listener.accept() => {
                    let (tcp, _addr) = match result {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!(error = %e, "TLS accept error");
                            continue;
                        }
                    };
                    let acceptor = acceptor.clone();
                    let router = router.clone();
                    let auto_builder = auto_builder.clone();
                    tokio::spawn(async move {
                        let tls_stream = match acceptor.accept(tcp).await {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::debug!(error = %e, "TLS handshake error");
                                return;
                            }
                        };

                        // Extract CN from client cert (present only for mTLS connections).
                        let cn = crate::tls::extract_cn_from_tls_stream(&tls_stream);

                        // Wrap the router with a per-connection layer that injects
                        // `MtlsClientCn` into request extensions.
                        let svc = if let Some(cn_value) = cn {
                            let cn_ext = crate::identity::MtlsClientCn(cn_value);
                            router.layer(axum::middleware::from_fn(
                                move |mut req: axum::extract::Request,
                                      next: axum::middleware::Next| {
                                    let cn_ext = cn_ext.clone();
                                    async move {
                                        req.extensions_mut().insert(cn_ext);
                                        next.run(req).await
                                    }
                                },
                            ))
                        } else {
                            // No client cert — pass through unchanged.
                            router
                        };

                        let io = TokioIo::new(tls_stream);
                        // axum::Router implements tower::Service, not hyper::Service
                        // directly — wrap it with TowerToHyperService.
                        let hyper_svc =
                            hyper_util::service::TowerToHyperService::new(svc);
                        if let Err(e) = auto_builder.serve_connection(io, hyper_svc).await {
                            tracing::debug!(error = %e, "TLS connection error");
                        }
                    });
                }
            }
        }
    } else {
        // Plain-TCP path (existing implementation).
        let listen_addr = listener.local_addr()?;
        tracing::info!(addr = %listen_addr, "listening");

        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                // Wait for the caller's shutdown signal (SIGTERM / Ctrl-C / test oneshot).
                shutdown.await;
                // Drain: keep existing connections alive until they finish or the
                // configured timeout expires, then close everything.
                tracing::info!(
                    drain_secs = shutdown_timeout.as_secs_f64(),
                    "graceful shutdown: draining in-flight requests"
                );
                tokio::time::sleep(shutdown_timeout).await;
            })
            .await?;
    }

    tracing::info!("server shut down cleanly");
    Ok(())
}
