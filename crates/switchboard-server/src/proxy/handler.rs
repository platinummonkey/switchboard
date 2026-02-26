//! Axum route handlers for the proxy endpoints.
//!
//! # Endpoints
//!
//! - `POST /v1/chat/completions` — OpenAI-compatible chat completions
//! - `POST /api/v1/messages` — Anthropic native Messages API
//! - `GET /v1/models` — List available models
//! - `GET /health` — Health check

use std::collections::HashMap;
use std::sync::Arc;

use axum::Extension;
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use switchboard_common::types::RequestContext;

use crate::auth::ValidatedClient;
use crate::config::IdentityConfig;
use crate::config::ServerConfig;
use crate::identity::{HeaderResolver, IdentityChain, JwtClaimResolver};
use crate::key_pool::KeyPool;
use crate::providers::ProviderRegistry;
use crate::proxy::error::ProxyError;
use crate::proxy::transform::{
    anthropic_to_proxied, openai_to_proxied, proxied_to_anthropic, proxied_to_openai,
};

// ── AppState ──────────────────────────────────────────────────────────────────

/// Shared application state available to all route handlers via [`State`].
#[derive(Clone)]
pub struct AppState {
    /// Full server configuration (hot-reloadable via [`crate::config::HotConfig`]).
    pub config: Arc<ServerConfig>,
    /// Registry of all registered upstream providers.
    pub providers: Arc<ProviderRegistry>,
    /// Key pools keyed by provider name (matches `config.providers` keys).
    pub key_pools: Arc<HashMap<String, Arc<KeyPool>>>,
}

// ── Route handlers ────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions` — OpenAI-compatible chat completions.
///
/// Parses the body as an OpenAI chat completion request, routes to the
/// appropriate upstream provider, and returns the response in OpenAI format.
/// If `stream: true` the response is forwarded as raw SSE.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Extension(validated_client): Extension<ValidatedClient>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let is_streaming = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let request = match openai_to_proxied(&body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };

    let model = request.model.clone();
    tracing::info!(model = %model, stream = is_streaming, "chat_completions request");

    let (provider, resolved_model) = match state
        .providers
        .resolve_provider(&model, &state.config.providers)
    {
        Some(p) => p,
        None => return ProxyError::NoProvider(model).into_response(),
    };

    let provider_name = provider.name().to_string();
    let key_pool = match state.key_pools.get(&provider_name) {
        Some(pool) => pool,
        None => return ProxyError::NoKey(provider_name).into_response(),
    };

    // Build RequestContext with resolved identity.
    let ctx = build_request_context(
        &validated_client,
        &headers,
        &state.config.identity,
        Some(resolved_model.clone()),
    )
    .await;

    let key = match key_pool.select(&ctx) {
        Some(k) => k,
        None => return ProxyError::NoKey(provider_name).into_response(),
    };

    if is_streaming {
        let stream_result = provider.send_streaming(request, key).await;
        match stream_result {
            Err(e) => {
                tracing::error!(error = %e, "streaming upstream error");
                ProxyError::Upstream(e.to_string()).into_response()
            }
            Ok(byte_stream) => {
                // Convert BoxStream<Bytes, SwitchboardError> to a reqwest-like
                // response using a channel-bridged synthetic reqwest::Response.
                // Since we can't easily create a reqwest::Response from a stream,
                // we build a streaming axum response directly.
                build_streaming_response(byte_stream).await
            }
        }
    } else {
        match provider.send(request, key).await {
            Err(e) => {
                tracing::error!(error = %e, "upstream error");
                ProxyError::Upstream(e.to_string()).into_response()
            }
            Ok(resp) => {
                let json = proxied_to_openai(&resp);
                Json(json).into_response()
            }
        }
    }
}

/// `POST /api/v1/messages` — Anthropic native Messages API.
///
/// Parses the body as an Anthropic Messages request, routes to the appropriate
/// upstream provider, and returns the response in Anthropic format.
pub async fn anthropic_messages(
    State(state): State<Arc<AppState>>,
    Extension(validated_client): Extension<ValidatedClient>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let is_streaming = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let request = match anthropic_to_proxied(&body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };

    let model = request.model.clone();
    tracing::info!(model = %model, stream = is_streaming, "anthropic_messages request");

    let (provider, resolved_model) = match state
        .providers
        .resolve_provider(&model, &state.config.providers)
    {
        Some(p) => p,
        None => return ProxyError::NoProvider(model).into_response(),
    };

    let provider_name = provider.name().to_string();
    let key_pool = match state.key_pools.get(&provider_name) {
        Some(pool) => pool,
        None => return ProxyError::NoKey(provider_name).into_response(),
    };

    // Build RequestContext with resolved identity.
    let ctx = build_request_context(
        &validated_client,
        &headers,
        &state.config.identity,
        Some(resolved_model.clone()),
    )
    .await;

    let key = match key_pool.select(&ctx) {
        Some(k) => k,
        None => return ProxyError::NoKey(provider_name).into_response(),
    };

    if is_streaming {
        let stream_result = provider.send_streaming(request, key).await;
        match stream_result {
            Err(e) => {
                tracing::error!(error = %e, "streaming upstream error");
                ProxyError::Upstream(e.to_string()).into_response()
            }
            Ok(byte_stream) => build_streaming_response(byte_stream).await,
        }
    } else {
        match provider.send(request, key).await {
            Err(e) => {
                tracing::error!(error = %e, "upstream error");
                ProxyError::Upstream(e.to_string()).into_response()
            }
            Ok(resp) => {
                let json = proxied_to_anthropic(&resp);
                Json(json).into_response()
            }
        }
    }
}

/// `GET /v1/models` — List all models available across registered providers.
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models: Vec<serde_json::Value> = state
        .config
        .providers
        .values()
        .flat_map(|cfg| {
            cfg.models.iter().map(|model| {
                serde_json::json!({
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "switchboard",
                })
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": models,
    }))
}

/// `GET /health` — Simple health check.
///
/// Returns `{"status": "ok"}` with HTTP 200.
pub async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

// ── Identity helpers ──────────────────────────────────────────────────────────

/// Build an [`IdentityChain`] from the server's identity config.
///
/// Currently supports `"header"` and `"jwt"` resolver strategies.
/// `"api_key"` and `"mtls_cn"` resolvers require runtime state and are
/// added to the chain in a future phase.
fn build_identity_chain(config: &IdentityConfig) -> IdentityChain {
    let mut resolvers: Vec<Box<dyn crate::identity::IdentityResolver>> = Vec::new();
    for resolver_name in &config.resolvers {
        match resolver_name.as_str() {
            "header" => resolvers.push(Box::new(HeaderResolver)),
            "jwt" => resolvers.push(Box::new(JwtClaimResolver::new(&config.jwt_claim))),
            _ => {
                tracing::debug!(resolver = %resolver_name, "identity resolver not yet wired");
            }
        }
    }
    IdentityChain::new(resolvers)
}

/// Build a [`RequestContext`] with identity resolved from the auth layer and
/// identity chain.
///
/// Steps:
/// 1. Start from a fresh context with a new request ID.
/// 2. Populate `user_id` from [`ValidatedClient`] (set by [`AuthLayer`]).
/// 3. Copy Switchboard protocol headers from the HTTP request headers.
/// 4. Run the [`IdentityChain`] to resolve / override user and team.
/// 5. Set the model.
async fn build_request_context(
    validated_client: &ValidatedClient,
    headers: &HeaderMap,
    identity_config: &IdentityConfig,
    model: Option<String>,
) -> RequestContext {
    let mut ctx = RequestContext::new();
    ctx.model = model;

    // Seed from ValidatedClient (JWT sub / email claim from AuthLayer).
    if let Some(uid) = &validated_client.user_id {
        ctx.user_id = Some(uid.clone());
    }

    // Copy Switchboard protocol headers into the context.
    for (name, value) in headers.iter() {
        if switchboard_common::protocol::is_switchboard_header(name.as_str()) {
            if let Ok(v) = value.to_str() {
                ctx.switchboard_headers
                    .insert(name.to_string(), v.to_string());
            }
        }
    }

    // Run the identity chain to resolve / override user_id and team.
    let chain = build_identity_chain(identity_config);
    let identity = chain.resolve(&ctx).await;
    if !identity.is_anonymous() {
        tracing::debug!(
            user_id = %identity.id,
            source = %identity.source,
            "identity resolved"
        );
        ctx.user_id = Some(identity.id.clone());
        ctx.team = identity.team.clone();
    }

    ctx
}

// ── Streaming response builder ────────────────────────────────────────────────

/// Build a streaming axum response from a [`crate::routing::BoxStream`].
///
/// Yields all bytes from the stream as-is, with SSE content headers set.
async fn build_streaming_response(mut stream: crate::routing::BoxStream) -> Response {
    use axum::body::Body;
    use futures_util::StreamExt;
    use switchboard_common::errors::SwitchboardError;

    let mapped = async_stream::stream! {
        while let Some(item) = stream.next().await {
            match item {
                Ok(bytes) => yield Ok::<_, std::io::Error>(bytes),
                Err(SwitchboardError::Upstream(e)) => {
                    tracing::warn!(error = %e, "stream error from upstream");
                    break;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "stream error");
                    break;
                }
            }
        }
    };

    let body = Body::from_stream(mapped);

    axum::http::Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .body(body)
        .unwrap_or_else(|_| {
            axum::http::Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::empty())
                .unwrap()
        })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::{get, post};
    use http::{HeaderName, HeaderValue};
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::auth::UpstreamCredentials;
    use crate::config::ServerConfig;
    use crate::config::provider::{KeyPoolConfig, ProviderConfig};
    use crate::key_pool::KeyHealth;
    use crate::key_pool::pool::KeyPool;
    use crate::key_pool::provider::{KeySource, PooledKey};
    use crate::key_pool::selector::WeightedRandomSelector;
    use crate::providers::{AnthropicProvider, OpenAiProvider, ProviderRegistry};

    fn make_key(header_name: &'static str, header_value: &'static str) -> PooledKey {
        PooledKey {
            id: "test-key".into(),
            credentials: UpstreamCredentials {
                header_name: HeaderName::from_static(header_name),
                header_value: HeaderValue::from_static(header_value),
                expires_at: None,
            },
            weight: 1.0,
            source: KeySource::Static,
            health: KeyHealth::default(),
        }
    }

    fn make_pool(key: PooledKey) -> Arc<KeyPool> {
        Arc::new(KeyPool::new(vec![key], Box::new(WeightedRandomSelector)))
    }

    fn openai_state(mock_url: &str) -> Arc<AppState> {
        let provider = Arc::new(OpenAiProvider::new_named(
            "openai",
            mock_url,
            vec!["gpt-4o".into()],
            Duration::from_secs(5),
        ));

        let mut registry = ProviderRegistry::new();
        registry.register("openai", provider);

        let mut key_pools = HashMap::new();
        key_pools.insert(
            "openai".into(),
            make_pool(make_key("authorization", "Bearer sk-test")),
        );

        let mut config = ServerConfig::default();
        let provider_config = ProviderConfig {
            base_url: Some(mock_url.into()),
            api_format: "openai".into(),
            models: vec!["gpt-4o".into()],
            region: None,
            cross_region_inference: false,
            project_id: None,
            timeout: "5s".into(),
            health_check_interval: "30s".into(),
            max_concurrent: 10,
            key_pool: KeyPoolConfig::default(),
        };
        config.providers.insert("openai".into(), provider_config);

        Arc::new(AppState {
            config: Arc::new(config),
            providers: Arc::new(registry),
            key_pools: Arc::new(key_pools),
        })
    }

    fn anthropic_state(mock_url: &str) -> Arc<AppState> {
        let provider = Arc::new(AnthropicProvider::new_with_base_url(
            mock_url,
            vec!["claude-sonnet-4-20250514".into()],
            Duration::from_secs(5),
        ));

        let mut registry = ProviderRegistry::new();
        registry.register("anthropic", provider);

        let mut key_pools = HashMap::new();
        key_pools.insert(
            "anthropic".into(),
            make_pool(make_key("x-api-key", "sk-ant-test")),
        );

        let mut config = ServerConfig::default();
        let provider_config = ProviderConfig {
            base_url: Some(mock_url.into()),
            api_format: "anthropic".into(),
            models: vec!["claude-sonnet-4-20250514".into()],
            region: None,
            cross_region_inference: false,
            project_id: None,
            timeout: "5s".into(),
            health_check_interval: "30s".into(),
            max_concurrent: 10,
            key_pool: KeyPoolConfig::default(),
        };
        config.providers.insert("anthropic".into(), provider_config);

        Arc::new(AppState {
            config: Arc::new(config),
            providers: Arc::new(registry),
            key_pools: Arc::new(key_pools),
        })
    }

    fn build_router(state: Arc<AppState>) -> Router {
        // Use a passthrough validated client extension.
        let validated_client = ValidatedClient::from_static_key();
        Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .route("/api/v1/messages", post(anthropic_messages))
            .route("/v1/models", get(list_models))
            .route("/health", get(health))
            .layer(axum::middleware::from_fn(
                move |mut req: axum::extract::Request, next: axum::middleware::Next| {
                    let vc = validated_client.clone();
                    async move {
                        req.extensions_mut().insert(vc);
                        next.run(req).await
                    }
                },
            ))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = Arc::new(AppState {
            config: Arc::new(ServerConfig::default()),
            providers: Arc::new(ProviderRegistry::new()),
            key_pools: Arc::new(HashMap::new()),
        });
        let app = build_router(state);

        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn test_list_models() {
        let mut config = ServerConfig::default();
        config.providers.insert(
            "openai".into(),
            ProviderConfig {
                base_url: Some("https://api.openai.com".into()),
                api_format: "openai".into(),
                models: vec!["gpt-4o".into(), "gpt-4o-mini".into()],
                region: None,
                cross_region_inference: false,
                project_id: None,
                timeout: "30s".into(),
                health_check_interval: "30s".into(),
                max_concurrent: 10,
                key_pool: KeyPoolConfig::default(),
            },
        );

        let state = Arc::new(AppState {
            config: Arc::new(config),
            providers: Arc::new(ProviderRegistry::new()),
            key_pools: Arc::new(HashMap::new()),
        });
        let app = build_router(state);

        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["object"], "list");
        let data = json["data"].as_array().unwrap();
        let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"gpt-4o"));
        assert!(ids.contains(&"gpt-4o-mini"));
    }

    #[tokio::test]
    async fn test_openai_proxy_non_streaming() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from mock!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
            })))
            .mount(&mock_server)
            .await;

        let state = openai_state(&mock_server.uri());
        let app = build_router(state);

        let request_body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["choices"][0]["message"]["content"], "Hello from mock!");
        assert_eq!(json["object"], "chat.completion");
    }

    #[tokio::test]
    async fn test_anthropic_proxy_non_streaming() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Anthropic mock response!"}],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 8, "output_tokens": 5}
            })))
            .mount(&mock_server)
            .await;

        let state = anthropic_state(&mock_server.uri());
        let app = build_router(state);

        let request_body = serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello Claude"}],
            "max_tokens": 256
        });

        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["content"][0]["text"], "Anthropic mock response!");
        assert_eq!(json["type"], "message");
    }

    #[tokio::test]
    async fn test_no_provider_returns_400() {
        let state = Arc::new(AppState {
            config: Arc::new(ServerConfig::default()),
            providers: Arc::new(ProviderRegistry::new()),
            key_pools: Arc::new(HashMap::new()),
        });
        let app = build_router(state);

        let request_body = serde_json::json!({
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hi"}]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_invalid_json_returns_422() {
        let state = Arc::new(AppState {
            config: Arc::new(ServerConfig::default()),
            providers: Arc::new(ProviderRegistry::new()),
            key_pools: Arc::new(HashMap::new()),
        });
        let app = build_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from("not valid json"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // axum returns 400 for malformed JSON that fails deserialization.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_upstream_error_returns_502() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&mock_server)
            .await;

        let state = openai_state(&mock_server.uri());
        let app = build_router(state);

        let request_body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    // ── Identity tests ────────────────────────────────────────────────────────

    #[test]
    fn test_build_identity_chain_with_header_resolver() {
        use crate::config::IdentityConfig;

        let config = IdentityConfig {
            resolvers: vec!["header".into()],
            header_name: "x-switchboard-user".into(),
            jwt_claim: "email".into(),
        };
        let chain = build_identity_chain(&config);
        // The chain is constructed with 1 resolver.
        let dbg = format!("{chain:?}");
        assert!(dbg.contains("resolver_count: 1"));
    }

    #[test]
    fn test_build_identity_chain_with_jwt_resolver() {
        use crate::config::IdentityConfig;

        let config = IdentityConfig {
            resolvers: vec!["jwt".into()],
            header_name: "x-switchboard-user".into(),
            jwt_claim: "email".into(),
        };
        let chain = build_identity_chain(&config);
        let dbg = format!("{chain:?}");
        assert!(dbg.contains("resolver_count: 1"));
    }

    #[test]
    fn test_build_identity_chain_unknown_resolver_skipped() {
        use crate::config::IdentityConfig;

        let config = IdentityConfig {
            resolvers: vec!["header".into(), "unknown_resolver".into(), "jwt".into()],
            header_name: "x-switchboard-user".into(),
            jwt_claim: "email".into(),
        };
        let chain = build_identity_chain(&config);
        // Only "header" and "jwt" are recognized; "unknown_resolver" is skipped.
        let dbg = format!("{chain:?}");
        assert!(dbg.contains("resolver_count: 2"));
    }

    #[tokio::test]
    async fn test_identity_extracted_from_validated_client() {
        use crate::auth::ValidatedClient;
        use crate::config::IdentityConfig;
        use axum::http::HeaderMap;

        let validated_client = ValidatedClient {
            user_id: Some("alice@example.com".into()),
            claims: HashMap::new(),
        };
        let headers = HeaderMap::new();
        let identity_config = IdentityConfig {
            resolvers: vec![], // no chain resolvers — rely on ValidatedClient
            header_name: "x-switchboard-user".into(),
            jwt_claim: "email".into(),
        };

        let ctx = build_request_context(
            &validated_client,
            &headers,
            &identity_config,
            Some("gpt-4o".into()),
        )
        .await;

        assert_eq!(ctx.user_id.as_deref(), Some("alice@example.com"));
        assert_eq!(ctx.model.as_deref(), Some("gpt-4o"));
    }

    #[tokio::test]
    async fn test_identity_chain_overrides_validated_client() {
        use crate::auth::ValidatedClient;
        use crate::config::IdentityConfig;
        use axum::http::{HeaderMap, HeaderName, HeaderValue};

        let validated_client = ValidatedClient {
            user_id: Some("jwt-user@example.com".into()),
            claims: HashMap::new(),
        };

        // Set X-Switchboard-User in the request headers — HeaderResolver should pick it up.
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-switchboard-user"),
            HeaderValue::from_static("header-user@example.com"),
        );

        let identity_config = IdentityConfig {
            resolvers: vec!["header".into()],
            header_name: "x-switchboard-user".into(),
            jwt_claim: "email".into(),
        };

        let ctx = build_request_context(&validated_client, &headers, &identity_config, None).await;

        // The header resolver should override the ValidatedClient user_id.
        assert_eq!(ctx.user_id.as_deref(), Some("header-user@example.com"));
    }

    #[tokio::test]
    async fn test_switchboard_headers_copied_to_context() {
        use crate::auth::ValidatedClient;
        use crate::config::IdentityConfig;
        use axum::http::{HeaderMap, HeaderName, HeaderValue};

        let validated_client = ValidatedClient::from_static_key();
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-switchboard-team"),
            HeaderValue::from_static("platform"),
        );
        headers.insert(
            HeaderName::from_static("x-switchboard-tool"),
            HeaderValue::from_static("claude-code"),
        );

        let identity_config = IdentityConfig::default();
        let ctx = build_request_context(&validated_client, &headers, &identity_config, None).await;

        assert_eq!(
            ctx.switchboard_headers
                .get("x-switchboard-team")
                .map(|s| s.as_str()),
            Some("platform")
        );
        assert_eq!(
            ctx.switchboard_headers
                .get("x-switchboard-tool")
                .map(|s| s.as_str()),
            Some("claude-code")
        );
    }
}
