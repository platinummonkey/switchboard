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

use switchboard_server::auth::UpstreamCredentials;
use switchboard_server::config::{self, ServerConfig};
use switchboard_server::key_pool::{
    KeyPool, KeySelector, LeastLoadedSelector, PooledKey, RoundRobinSelector,
    WeightedRandomSelector,
};
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

    let listen_addr = server_config.server.listen.clone();

    // Build provider registry and key pools from config.
    let (provider_registry, key_pools) = build_providers(&server_config);

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
    axum::serve(listener, router).await?;

    Ok(())
}

/// Build the [`ProviderRegistry`] and key pools from the server config.
fn build_providers(config: &ServerConfig) -> (ProviderRegistry, HashMap<String, Arc<KeyPool>>) {
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
        let keys: Vec<PooledKey> = provider_cfg
            .key_pool
            .keys
            .iter()
            .filter_map(|entry| {
                if entry.key_type != "static" {
                    tracing::warn!(
                        key_id = %entry.id,
                        key_type = %entry.key_type,
                        "only static keys supported in this phase, skipping"
                    );
                    return None;
                }
                let api_key = entry.api_key.as_deref().unwrap_or("");
                let (header_name, header_value_str) = match provider_cfg.api_format.as_str() {
                    "anthropic" => (
                        http::HeaderName::from_static("x-api-key"),
                        api_key.to_string(),
                    ),
                    "bedrock" => {
                        // For Bedrock, the api_key field holds a JSON credentials
                        // blob passed verbatim as the x-switchboard-bedrock-creds header.
                        (
                            http::HeaderName::from_static("x-switchboard-bedrock-creds"),
                            api_key.to_string(),
                        )
                    }
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
                Some(PooledKey::new_static(&entry.id, creds, entry.weight))
            })
            .collect();

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
