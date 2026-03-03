//! Admin control plane — REST API, authentication, and server.
//!
//! The admin server listens on a separate port (default `127.0.0.1:9090`)
//! and exposes all management operations via a JSON REST API under
//! `/admin/api/v1/`.

pub mod api;
pub mod auth;
pub mod ui;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::config::HotConfig;
use crate::db::DbPool;
use crate::error::ServerError;
use crate::key_pool::KeyPool;
use crate::middleware::RateLimitHandle;
use crate::observability::UsageTracker;

pub use auth::AdminAuthState;

// ── AdminState ────────────────────────────────────────────────────────────────

/// Shared state for the admin server.
///
/// Holds references to the hot-reloadable config, the key pools (wrapped in
/// `RwLock` so the admin API can mutate them at runtime), the auth state, the
/// shared in-memory usage tracker, and the live rate-limit handle for instant
/// override updates.
pub struct AdminState {
    /// Hot-reloadable server configuration.
    pub hot_config: Arc<HotConfig>,

    /// Key pools keyed by provider name. Each pool is guarded by a `RwLock`
    /// so the admin API can add/remove/update keys without restarting.
    pub key_pools: Arc<HashMap<String, Arc<RwLock<KeyPool>>>>,

    /// Admin authentication state (static token or JWT config).
    pub auth_state: Arc<AdminAuthState>,

    /// Shared in-memory usage tracker (same instance as the proxy handler's).
    pub usage: Arc<UsageTracker>,

    /// Handle to the live rate-limit override table.  Updates here take effect
    /// immediately without restarting the server.
    pub rate_limit_handle: RateLimitHandle,

    /// Optional Postgres connection pool.  `None` when `database.enabled = false`.
    pub db_pool: Option<Arc<DbPool>>,
}

impl AdminState {
    /// Create a new [`AdminState`].
    pub fn new(
        hot_config: Arc<HotConfig>,
        key_pools: Arc<HashMap<String, Arc<RwLock<KeyPool>>>>,
        auth_state: Arc<AdminAuthState>,
        usage: Arc<UsageTracker>,
        rate_limit_handle: RateLimitHandle,
        db_pool: Option<Arc<DbPool>>,
    ) -> Self {
        Self {
            hot_config,
            key_pools,
            auth_state,
            usage,
            rate_limit_handle,
            db_pool,
        }
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

/// Build the admin [`axum::Router`] with all routes and the auth extension.
///
/// The [`AdminAuthState`] is inserted as a request extension so the
/// [`auth::AdminAuth`] extractor can access it in every handler.
pub fn admin_router(state: Arc<AdminState>) -> axum::Router {
    use axum::middleware;

    let auth_state = Arc::clone(&state.auth_state);
    let router = api::admin_api_router(state);

    // Mount the embedded admin UI (feature-gated).
    #[cfg(feature = "admin-ui")]
    let router = router
        .route("/admin/", axum::routing::get(ui::serve_ui))
        .route("/admin/{*path}", axum::routing::get(ui::serve_ui));

    // Inject AdminAuthState as an extension on every request.
    router.layer(middleware::from_fn(
        move |mut req: axum::extract::Request, next: axum::middleware::Next| {
            let auth = Arc::clone(&auth_state);
            async move {
                req.extensions_mut().insert(auth);
                next.run(req).await
            }
        },
    ))
}

// ── Serve ─────────────────────────────────────────────────────────────────────

/// Start the admin server on an already-bound [`tokio::net::TcpListener`].
///
/// Useful for tests that need to know the admin server port before starting
/// (bind to `127.0.0.1:0`, read the port, then pass the listener here).
pub async fn serve_admin_on_listener(
    state: Arc<AdminState>,
    listener: tokio::net::TcpListener,
) -> Result<(), ServerError> {
    let router = admin_router(state);
    tracing::info!(addr = %listener.local_addr().unwrap_or_else(|_| "unknown".parse().unwrap()), "admin server listening");
    axum::serve(listener, router)
        .await
        .map_err(ServerError::Io)?;
    Ok(())
}

/// Start the admin server and bind it to `listen`.
///
/// This function runs until the server is stopped (e.g., via a signal or
/// the handle being dropped). Typically called inside `tokio::spawn`.
pub async fn serve_admin(state: Arc<AdminState>, listen: &str) -> Result<(), ServerError> {
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .map_err(ServerError::Io)?;
    serve_admin_on_listener(state, listener).await
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use super::*;
    use crate::admin::AdminAuthState;
    use crate::config::{AdminConfig, HotConfig, ServerConfig};
    use crate::key_pool::{KeyPool, WeightedRandomSelector};
    use crate::middleware::RateLimitLayer;

    fn make_hot_config() -> Arc<HotConfig> {
        let cfg = ServerConfig::default();
        // Use a non-existent path — tests that call `reload` will see an error.
        Arc::new(HotConfig::new(cfg, "/tmp/nonexistent-test-config.toml"))
    }

    fn make_admin_state_with_token(token: &str) -> Arc<AdminState> {
        let hot_config = make_hot_config();
        let pools = Arc::new(HashMap::new());
        let admin_config = AdminConfig {
            enabled: true,
            auth: "static_token".into(),
            static_token: Some(token.into()),
            ..AdminConfig::default()
        };
        let auth_state = Arc::new(AdminAuthState::new(admin_config));
        let (_layer, handle) = RateLimitLayer::new(60, 100_000, std::iter::empty());
        Arc::new(AdminState::new(
            hot_config,
            pools,
            auth_state,
            Arc::new(crate::observability::UsageTracker::new()),
            handle,
            None,
        ))
    }

    fn build_router(state: Arc<AdminState>) -> axum::Router {
        admin_router(state)
    }

    // ── /health ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_health_returns_ok() {
        let state = make_admin_state_with_token("test-token");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["status"], "ok");
    }

    // ── Auth ───────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_auth_required() {
        let state = make_admin_state_with_token("secret-token");
        let app = build_router(state);

        // No Authorization header.
        let req = Request::builder()
            .uri("/admin/api/v1/config")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_admin_auth_static_token_correct() {
        let state = make_admin_state_with_token("my-token");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/config")
            .header("Authorization", "Bearer my-token")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_admin_auth_wrong_token() {
        let state = make_admin_state_with_token("correct-token");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/config")
            .header("Authorization", "Bearer wrong-token")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ── /providers ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_list_providers_empty() {
        let state = make_admin_state_with_token("tok");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/providers")
            .header("Authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json.is_array());
    }

    #[tokio::test]
    async fn test_admin_list_providers_with_pool() {
        let mut config = ServerConfig::default();
        config.providers.insert(
            "openai".into(),
            crate::config::provider::ProviderConfig {
                base_url: Some("https://api.openai.com".into()),
                api_format: "openai".into(),
                models: vec!["gpt-4o".into()],
                region: None,
                cross_region_inference: false,
                project_id: None,
                timeout: "30s".into(),
                health_check_interval: "30s".into(),
                max_concurrent: 10,
                key_pool: crate::config::provider::KeyPoolConfig::default(),
            },
        );

        let hot_config = Arc::new(HotConfig::new(config, "/tmp/nonexistent.toml"));
        let pool = Arc::new(RwLock::new(KeyPool::new(
            vec![],
            Box::new(WeightedRandomSelector),
        )));
        let mut pools = HashMap::new();
        pools.insert("openai".into(), pool);

        let admin_config = AdminConfig {
            enabled: true,
            auth: "static_token".into(),
            static_token: Some("tok".into()),
            ..AdminConfig::default()
        };
        let auth_state = Arc::new(AdminAuthState::new(admin_config));
        let (_layer, handle) = RateLimitLayer::new(60, 100_000, std::iter::empty());
        let state = Arc::new(AdminState::new(
            hot_config,
            Arc::new(pools),
            auth_state,
            Arc::new(crate::observability::UsageTracker::new()),
            handle,
            None,
        ));
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/providers")
            .header("Authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "openai");
    }

    // ── /model-selection ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_get_model_selection() {
        let state = make_admin_state_with_token("tok");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/model-selection")
            .header("Authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["mode"].is_string());
    }

    #[tokio::test]
    async fn test_admin_put_model_selection() {
        let state = make_admin_state_with_token("tok");
        let app = build_router(Arc::clone(&state));

        let new_config = serde_json::json!({
            "mode": "static",
            "model": "gpt-4o",
            "header": "x-switchboard-model",
            "fallback": null,
            "mappings": {},
            "allowed_models": [],
            "overrides": {}
        });

        let req = Request::builder()
            .method("PUT")
            .uri("/admin/api/v1/model-selection")
            .header("Authorization", "Bearer tok")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&new_config).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify the update is reflected.
        let cfg = state.hot_config.load();
        assert_eq!(cfg.model_selection.mode, "static");
        assert_eq!(cfg.model_selection.model.as_deref(), Some("gpt-4o"));
    }

    // ── /guardrails ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_get_guardrails() {
        let state = make_admin_state_with_token("tok");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/guardrails")
            .header("Authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["enabled"].is_boolean());
    }

    // ── /config/reload ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_config_reload_missing_file() {
        let state = make_admin_state_with_token("tok");
        let app = build_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/admin/api/v1/config/reload")
            .header("Authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Config file doesn't exist → should return 500.
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["error"].is_string());
    }

    // ── /config (redaction) ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_get_config_redacts_secrets() {
        let mut config = ServerConfig::default();
        config.admin.static_token = Some("ultra-secret-token".into());
        config.providers.insert(
            "anthropic".into(),
            crate::config::provider::ProviderConfig {
                base_url: Some("https://api.anthropic.com".into()),
                api_format: "anthropic".into(),
                models: vec!["claude-sonnet-4-20250514".into()],
                region: None,
                cross_region_inference: false,
                project_id: None,
                timeout: "30s".into(),
                health_check_interval: "30s".into(),
                max_concurrent: 10,
                key_pool: {
                    let mut kp = crate::config::provider::KeyPoolConfig::default();
                    kp.keys.push(crate::config::provider::KeyEntry {
                        id: "k1".into(),
                        key_type: "static".into(),
                        api_key: Some("sk-ant-secret".into()),
                        role_arn: None,
                        region: None,
                        refresh_interval: None,
                        vault_path: None,
                        weight: 1.0,
                    });
                    kp
                },
            },
        );

        let hot_config = Arc::new(HotConfig::new(config, "/tmp/nonexistent.toml"));
        let admin_config = AdminConfig {
            enabled: true,
            auth: "static_token".into(),
            static_token: Some("admin-secret".into()),
            ..AdminConfig::default()
        };
        let auth_state = Arc::new(AdminAuthState::new(admin_config));
        let (_layer, handle) = RateLimitLayer::new(60, 100_000, std::iter::empty());
        let state = Arc::new(AdminState::new(
            hot_config,
            Arc::new(HashMap::new()),
            auth_state,
            Arc::new(crate::observability::UsageTracker::new()),
            handle,
            None,
        ));
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/config")
            .header("Authorization", "Bearer admin-secret")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 16384).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        // admin.static_token should be redacted.
        assert_eq!(json["admin"]["static_token"], "<redacted>");

        // Provider api_key should be redacted.
        let api_key = &json["providers"]["anthropic"]["key_pool"]["keys"][0]["api_key"];
        assert_eq!(api_key, "<redacted>");
    }

    // ── /rate-limits ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_get_rate_limits() {
        let state = make_admin_state_with_token("tok");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/rate-limits")
            .header("Authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── /routing/semantic ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_get_semantic_routing() {
        let state = make_admin_state_with_token("tok");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/routing/semantic")
            .header("Authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── /users and /usage ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_admin_list_users_stub() {
        let state = make_admin_state_with_token("tok");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/users")
            .header("Authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["total"], 0);
    }

    #[tokio::test]
    async fn test_admin_get_usage_stub() {
        let state = make_admin_state_with_token("tok");
        let app = build_router(state);

        let req = Request::builder()
            .uri("/admin/api/v1/usage")
            .header("Authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["total_requests"], 0);
    }
}
