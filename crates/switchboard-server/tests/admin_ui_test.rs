//! Integration tests for the embedded admin UI (Phase 14).
//!
//! These tests verify that static SPA assets are correctly embedded and served
//! by the admin router when the `admin-ui` feature is enabled, and that the
//! fallback stub returns 404 when the feature is absent.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use switchboard_server::admin::{AdminAuthState, AdminState, admin_router};
use switchboard_server::config::{AdminConfig, HotConfig, ServerConfig};
use switchboard_server::key_pool::KeyPool;

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_admin_state() -> Arc<AdminState> {
    let hot_config = Arc::new(HotConfig::new(
        ServerConfig::default(),
        "/tmp/nonexistent-ui-test.toml",
    ));
    let pools: Arc<HashMap<String, Arc<std::sync::RwLock<KeyPool>>>> = Arc::new(HashMap::new());
    let admin_config = AdminConfig {
        enabled: true,
        auth: "static_token".into(),
        static_token: Some("test-token".into()),
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

// ── Asset embedding tests (require admin-ui feature) ─────────────────────────

/// Verify that `index.html` is embedded and contains the expected page title.
#[cfg(feature = "admin-ui")]
#[test]
fn test_ui_assets_embedded() {
    use switchboard_server::admin::ui::ui_impl::AdminUiAssets;

    let asset = AdminUiAssets::get("index.html");
    assert!(asset.is_some(), "index.html must be embedded in the binary");

    let bytes = asset.unwrap().data;
    let text = std::str::from_utf8(&bytes).expect("index.html must be valid UTF-8");
    assert!(
        text.contains("Switchboard Admin"),
        "index.html must contain 'Switchboard Admin'; got: {text:.200}"
    );
}

// ── HTTP serving tests ────────────────────────────────────────────────────────

/// `GET /admin/` must return 200 with `text/html` content-type.
#[cfg(feature = "admin-ui")]
#[tokio::test]
async fn test_ui_serve_index() {
    let state = make_admin_state();
    let app = admin_router(state);

    let req = Request::builder()
        .uri("/admin/")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "GET /admin/ must return 200");

    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("text/html"),
        "Content-Type must be text/html; got: {ct}"
    );
}

/// `GET /admin/app.js` must return 200 with a JavaScript content-type.
#[cfg(feature = "admin-ui")]
#[tokio::test]
async fn test_ui_serve_js() {
    let state = make_admin_state();
    let app = admin_router(state);

    let req = Request::builder()
        .uri("/admin/app.js")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /admin/app.js must return 200"
    );

    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("javascript"),
        "Content-Type must contain 'javascript'; got: {ct}"
    );
}

/// `GET /admin/style.css` must return 200 with a CSS content-type.
#[cfg(feature = "admin-ui")]
#[tokio::test]
async fn test_ui_serve_css() {
    let state = make_admin_state();
    let app = admin_router(state);

    let req = Request::builder()
        .uri("/admin/style.css")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /admin/style.css must return 200"
    );

    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("css"),
        "Content-Type must contain 'css'; got: {ct}"
    );
}

/// `GET /admin/nonexistent.xyz` must return 404.
#[cfg(feature = "admin-ui")]
#[tokio::test]
async fn test_ui_404_on_missing_asset() {
    let state = make_admin_state();
    let app = admin_router(state);

    let req = Request::builder()
        .uri("/admin/nonexistent.xyz")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "GET /admin/nonexistent.xyz must return 404"
    );
}

// ── Disabled-feature stub test ────────────────────────────────────────────────

/// When the `admin-ui` feature is disabled, `serve_ui` returns 404.
///
/// We can't toggle features at test time, so this test directly calls the
/// `#[cfg(not(feature = "admin-ui"))]` stub by calling through the router
/// which will route to the correct implementation at compile time.
/// Under `--all-features` the enabled path is exercised by the tests above.
/// This test validates the contract documented on the stub.
#[cfg(not(feature = "admin-ui"))]
#[tokio::test]
async fn test_ui_disabled_without_feature() {
    use axum::http::Uri;
    use axum::response::IntoResponse;
    use switchboard_server::admin::ui::serve_ui;

    let uri: Uri = "/admin/".parse().unwrap();
    let resp = serve_ui(uri).await.into_response();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "serve_ui stub must return 404 when admin-ui feature is disabled"
    );
}
