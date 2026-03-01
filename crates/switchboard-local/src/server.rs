//! Local HTTP server for `switchboard-local`.
//!
//! [`LocalServer`] listens on `localhost:8877` (configurable) and exposes
//! OpenAI-compatible and Anthropic-native endpoints.  All proxied routes
//! delegate to [`crate::proxy::forward_request`].

use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use serde_json::json;
use tracing::instrument;

use crate::auth::LocalAuthManager;
use crate::config::LocalConfig;
use crate::error::LocalError;
use crate::model_prefs::ModelPrefs;
use crate::proxy::{LocalServerState, forward_request};

/// The local proxy server.
pub struct LocalServer {
    config: Arc<LocalConfig>,
    auth_manager: Arc<LocalAuthManager>,
    model_prefs: Arc<ModelPrefs>,
    client: reqwest::Client,
}

impl LocalServer {
    /// Construct a new [`LocalServer`] from the given configuration.
    ///
    /// Creates the [`LocalAuthManager`] (which may spawn a background JWT
    /// refresh task) and the shared `reqwest` client.
    pub async fn new(config: Arc<LocalConfig>) -> Result<Self, LocalError> {
        let auth_manager = LocalAuthManager::new(&config.auth).await?;
        let model_prefs = ModelPrefs::from_config(&config.model);

        // For mTLS, `build_client()` returns the TLS-configured client (with
        // the embedded client certificate + trusted server CA).  For all other
        // auth methods it returns a plain default client; auth is carried via
        // the `Authorization` header injected in `forward_request`.
        let client = auth_manager.build_client();

        Ok(Self {
            config,
            auth_manager: Arc::new(auth_manager),
            model_prefs: Arc::new(model_prefs),
            client,
        })
    }

    /// Start the local proxy server and block until it shuts down.
    pub async fn run(self) -> Result<(), LocalError> {
        let listen_addr: std::net::SocketAddr = self
            .config
            .local
            .listen
            .parse()
            .map_err(|e| LocalError::Config(format!("invalid listen address: {e}")))?;

        let state = Arc::new(LocalServerState {
            config: Arc::clone(&self.config),
            auth_manager: Arc::clone(&self.auth_manager),
            model_prefs: Arc::clone(&self.model_prefs),
            client: self.client,
        });

        let router = build_router(state);

        tracing::info!(addr = %listen_addr, "switchboard-local listening");

        let listener = tokio::net::TcpListener::bind(listen_addr)
            .await
            .map_err(LocalError::Io)?;

        axum::serve(listener, router)
            .await
            .map_err(LocalError::Io)?;

        Ok(())
    }
}

/// Build the axum [`Router`] for `switchboard-local`.
pub fn build_router(state: Arc<LocalServerState>) -> Router {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(proxy_handler))
        .route("/v1/completions", post(proxy_handler))
        .route("/v1/embeddings", post(proxy_handler))
        .route("/v1/models", get(proxy_handler))
        // Anthropic-native endpoint
        .route("/api/v1/messages", post(proxy_handler))
        // Local health check
        .route("/health", get(health_handler))
        .with_state(state)
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// Generic proxy handler: extracts the method, URI path, headers, and body
/// from the request and delegates to [`forward_request`].
#[instrument(skip_all)]
async fn proxy_handler(State(state): State<Arc<LocalServerState>>, request: Request) -> Response {
    let method: Method = request.method().clone();
    let path = request
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_owned())
        .unwrap_or_else(|| request.uri().path().to_owned());
    let headers: HeaderMap = request.headers().clone();

    let body_bytes: Bytes = match axum::body::to_bytes(request.into_body(), usize::MAX).await {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(error = %e, "local proxy: failed to read request body");
            return axum::http::StatusCode::BAD_REQUEST.into_response();
        }
    };

    forward_request(state, method, &path, headers, body_bytes).await
}

/// Health check endpoint: returns `200 OK` with JSON status.
async fn health_handler(State(state): State<Arc<LocalServerState>>) -> impl IntoResponse {
    let auth_method = state.config.auth.method.clone();
    let server_url = state.config.server.url.clone();

    Json(json!({
        "status": "ok",
        "server": server_url,
        "auth": auth_method,
    }))
}
