//! switchboard-server binary entry point.
//!
//! Loads configuration, builds providers and key pools, and starts the axum
//! HTTP server.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use axum::routing::{get, post};
use http::HeaderValue;
use tokio::signal;

use switchboard_server::auth::UpstreamCredentials;
use switchboard_server::config::{self, ServerConfig};
use switchboard_server::key_pool::{
    AwsStsProvider, KeyPool, KeyProvider, KeySelector, LeastLoadedSelector, PooledKey,
    RoundRobinSelector, WeightedRandomSelector,
};
use switchboard_server::observability;
use switchboard_server::providers::{
    AnthropicProvider, BedrockProvider, OllamaProvider, OpenAiProvider, ProviderRegistry,
    VertexProvider,
};
use switchboard_server::proxy::handler::{
    AppState, anthropic_messages, chat_completions, health, list_models,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("switchboard-server starting");

    // Load config from SWITCHBOARD_CONFIG env var, default to
    // config/switchboard-server.toml.
    let config_path = std::env::var("SWITCHBOARD_CONFIG")
        .unwrap_or_else(|_| "config/switchboard-server.toml".to_string());

    let server_config = if std::path::Path::new(&config_path).exists() {
        tracing::info!(path = %config_path, "loading config");
        config::load(std::path::Path::new(&config_path))?
    } else {
        tracing::warn!(
            path = %config_path,
            "config file not found, using defaults"
        );
        ServerConfig::default()
    };

    // Initialise OTel tracing (no-op when observability.enabled = false).
    let _otel_guard = observability::init_tracing(&server_config.observability)?;

    let listen_addr = server_config.server.listen.clone();

    // Extract fields needed after config is moved into AppState.
    let shutdown_timeout = switchboard_server::config::duration::parse(
        &server_config.server.graceful_shutdown_timeout,
    )
    .unwrap_or(std::time::Duration::from_secs(30));

    // Build provider registry and key pools from config.
    let (provider_registry, key_pools) = build_providers(&server_config).await;

    let app_state = Arc::new(AppState {
        config: Arc::new(server_config),
        providers: Arc::new(provider_registry),
        key_pools: Arc::new(key_pools),
    });

    let router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/api/v1/messages", post(anthropic_messages))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(app_state);

    tracing::info!(addr = %listen_addr, "listening");
    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal(shutdown_timeout))
        .await?;

    tracing::info!("server shut down cleanly");
    Ok(())
}

/// Returns a future that resolves when SIGTERM or SIGINT is received,
/// then waits an additional `drain` period for in-flight requests to finish.
async fn shutdown_signal(drain: std::time::Duration) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let sigterm = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let sigterm = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c  => tracing::info!("received Ctrl-C"),
        _ = sigterm => tracing::info!("received SIGTERM"),
    }

    tracing::info!(
        drain_secs = drain.as_secs_f64(),
        "draining in-flight requests"
    );
    tokio::time::sleep(drain).await;
}

/// Build the [`ProviderRegistry`] and key pools from the server config.
async fn build_providers(
    config: &ServerConfig,
) -> (ProviderRegistry, HashMap<String, Arc<KeyPool>>) {
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
                                source: switchboard_server::key_pool::KeySource::AwsSts {
                                    role_arn: role_arn.to_string(),
                                },
                                health: switchboard_server::key_pool::KeyHealth::default(),
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
