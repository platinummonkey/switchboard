//! Integration tests for the Admin REST API.
//!
//! Each test builds an `AdminState`, constructs the admin axum router, and
//! makes HTTP requests through the full handler stack using `tower::ServiceExt`.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use switchboard_server::admin::{AdminAuthState, AdminState, admin_router};
use switchboard_server::config::provider::{KeyEntry, KeyPoolConfig, ProviderConfig};
use switchboard_server::config::{AdminConfig, HotConfig, ServerConfig};
use switchboard_server::key_pool::{KeyPool, WeightedRandomSelector};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_hot_config_default() -> Arc<HotConfig> {
    Arc::new(HotConfig::new(
        ServerConfig::default(),
        "/tmp/nonexistent-admin-test.toml",
    ))
}

fn make_hot_config_with_providers() -> Arc<HotConfig> {
    let mut config = ServerConfig::default();
    config.providers.insert(
        "anthropic".into(),
        ProviderConfig {
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
                let mut kp = KeyPoolConfig::default();
                kp.keys.push(KeyEntry {
                    id: "k1".into(),
                    key_type: "static".into(),
                    api_key: Some("sk-ant-supersecret".into()),
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
    Arc::new(HotConfig::new(config, "/tmp/nonexistent-admin-test.toml"))
}

fn make_admin_state(token: &str) -> Arc<AdminState> {
    make_admin_state_with_config(make_hot_config_default(), token)
}

fn make_admin_state_with_config(hot_config: Arc<HotConfig>, token: &str) -> Arc<AdminState> {
    let pools = Arc::new(HashMap::new());
    let admin_config = AdminConfig {
        enabled: true,
        auth: "static_token".into(),
        static_token: Some(token.into()),
        ..AdminConfig::default()
    };
    let auth_state = Arc::new(AdminAuthState::new(admin_config));
    Arc::new(AdminState::new(
        hot_config,
        pools,
        auth_state,
        Arc::new(switchboard_server::observability::UsageTracker::new()),
    ))
}

fn make_admin_state_with_pool(token: &str) -> Arc<AdminState> {
    let hot_config = make_hot_config_with_providers();
    let pool = Arc::new(RwLock::new(KeyPool::new(
        vec![],
        Box::new(WeightedRandomSelector),
    )));
    let mut pools = HashMap::new();
    pools.insert("anthropic".into(), pool);
    let admin_config = AdminConfig {
        enabled: true,
        auth: "static_token".into(),
        static_token: Some(token.into()),
        ..AdminConfig::default()
    };
    let auth_state = Arc::new(AdminAuthState::new(admin_config));
    Arc::new(AdminState::new(
        hot_config,
        Arc::new(pools),
        auth_state,
        Arc::new(switchboard_server::observability::UsageTracker::new()),
    ))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_admin_server_starts_when_enabled() {
    // Build an AdminState with a random port and static-token auth.
    // Verify that serve_admin binds successfully.
    use switchboard_server::admin::serve_admin;

    let admin_config = AdminConfig {
        enabled: true,
        auth: "static_token".into(),
        static_token: Some("test-token".into()),
        listen: "127.0.0.1:0".into(),
        ..AdminConfig::default()
    };

    let server_config = ServerConfig {
        admin: AdminConfig {
            enabled: true,
            auth: "static_token".into(),
            static_token: Some("test-token".into()),
            ..AdminConfig::default()
        },
        ..ServerConfig::default()
    };

    let auth_state = Arc::new(AdminAuthState::new(admin_config));
    let hot_config = Arc::new(HotConfig::new(server_config, "/nonexistent/config.toml"));
    let state = Arc::new(AdminState::new(
        hot_config,
        Arc::new(HashMap::new()),
        auth_state,
        Arc::new(switchboard_server::observability::UsageTracker::new()),
    ));

    // serve_admin with port 0 should bind successfully.
    // We just verify it doesn't immediately error.
    let handle = tokio::spawn(async move { serve_admin(state, "127.0.0.1:0").await });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(!handle.is_finished() || handle.await.unwrap().is_ok());
}

#[tokio::test]
async fn test_admin_health_returns_ok() {
    let state = make_admin_state("tok");
    let app = admin_router(state);

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

#[tokio::test]
async fn test_admin_auth_required() {
    let state = make_admin_state("secret-token");
    let app = admin_router(state);

    // No Authorization header.
    let req = Request::builder()
        .uri("/admin/api/v1/config")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["error"].is_string());
}

#[tokio::test]
async fn test_admin_auth_static_token() {
    let state = make_admin_state("valid-token");
    let app = admin_router(state);

    let req = Request::builder()
        .uri("/admin/api/v1/config")
        .header("Authorization", "Bearer valid-token")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_admin_auth_wrong_token() {
    let state = make_admin_state("correct-token");
    let app = admin_router(state);

    let req = Request::builder()
        .uri("/admin/api/v1/config")
        .header("Authorization", "Bearer wrong-token")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["error"].is_string());
}

#[tokio::test]
async fn test_admin_list_providers() {
    let state = make_admin_state_with_pool("tok");
    let app = admin_router(state);

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

    let arr = json.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["id"], "anthropic");
    assert_eq!(arr[0]["api_format"], "anthropic");
}

#[tokio::test]
async fn test_admin_get_model_selection() {
    let state = make_admin_state("tok");
    let app = admin_router(state);

    let req = Request::builder()
        .uri("/admin/api/v1/model-selection")
        .header("Authorization", "Bearer tok")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Should have a "mode" field.
    assert!(json["mode"].is_string());
}

#[tokio::test]
async fn test_admin_put_model_selection() {
    let hot_config = make_hot_config_default();
    let state = make_admin_state_with_config(Arc::clone(&hot_config), "tok");
    let app = admin_router(Arc::clone(&state));

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

    // Verify the update is reflected in the hot config.
    let cfg = state.hot_config.load();
    assert_eq!(cfg.model_selection.mode, "static");
    assert_eq!(cfg.model_selection.model.as_deref(), Some("gpt-4o"));
}

#[tokio::test]
async fn test_admin_get_guardrails() {
    let state = make_admin_state("tok");
    let app = admin_router(state);

    let req = Request::builder()
        .uri("/admin/api/v1/guardrails")
        .header("Authorization", "Bearer tok")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Default config has enabled = false.
    assert_eq!(json["enabled"], false);
}

#[tokio::test]
async fn test_admin_config_reload_missing_file() {
    let state = make_admin_state("tok");
    let app = admin_router(state);

    let req = Request::builder()
        .method("POST")
        .uri("/admin/api/v1/config/reload")
        .header("Authorization", "Bearer tok")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // Config file doesn't exist → 500.
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["error"].is_string());
    let err = json["error"].as_str().unwrap();
    assert!(!err.is_empty());
}

#[tokio::test]
async fn test_admin_get_config_redacts_secrets() {
    // Build a config with a real secret in static_token and api_key.
    let hot_config = make_hot_config_with_providers();
    // The hot_config includes providers with api_key; also set admin token.
    {
        hot_config.update(|cfg| {
            cfg.admin.static_token = Some("ultra-secret-admin-token".into());
        });
    }

    let admin_config = AdminConfig {
        enabled: true,
        auth: "static_token".into(),
        static_token: Some("ultra-secret-admin-token".into()),
        ..AdminConfig::default()
    };
    let auth_state = Arc::new(AdminAuthState::new(admin_config));
    let state = Arc::new(AdminState::new(
        hot_config,
        Arc::new(HashMap::new()),
        auth_state,
        Arc::new(switchboard_server::observability::UsageTracker::new()),
    ));
    let app = admin_router(state);

    let req = Request::builder()
        .uri("/admin/api/v1/config")
        .header("Authorization", "Bearer ultra-secret-admin-token")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    // admin.static_token must be redacted.
    assert_eq!(json["admin"]["static_token"], "<redacted>");

    // Provider api_key must be redacted.
    let api_key = &json["providers"]["anthropic"]["key_pool"]["keys"][0]["api_key"];
    assert_eq!(api_key, "<redacted>");

    // Non-secret fields must still be present.
    assert_eq!(json["providers"]["anthropic"]["api_format"], "anthropic");
}
