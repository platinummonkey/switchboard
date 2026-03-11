//! Admin UI — embedded static SPA served under `/admin/`.
//!
//! When the `admin-ui` feature is enabled the three assets from `admin-ui/`
//! are embedded at compile time via `include_bytes!`. The [`serve_ui`] handler
//! looks up the requested path and responds with the correct `Content-Type`.
//!
//! When the feature is disabled every request to `/admin/*` returns 404.

// ── Feature-gated implementation ─────────────────────────────────────────────

#[cfg(feature = "admin-ui")]
pub mod ui_impl {
    pub struct EmbeddedFile {
        pub data: &'static [u8],
        pub content_type: &'static str,
    }

    pub struct AdminUiAssets;

    impl AdminUiAssets {
        pub fn get(path: &str) -> Option<EmbeddedFile> {
            match path {
                "index.html" => Some(EmbeddedFile {
                    data: include_bytes!("../../../../admin-ui/index.html"),
                    content_type: "text/html; charset=utf-8",
                }),
                "app.js" => Some(EmbeddedFile {
                    data: include_bytes!("../../../../admin-ui/app.js"),
                    content_type: "application/javascript; charset=utf-8",
                }),
                "style.css" => Some(EmbeddedFile {
                    data: include_bytes!("../../../../admin-ui/style.css"),
                    content_type: "text/css; charset=utf-8",
                }),
                _ => None,
            }
        }
    }
}

// ── Public handler ────────────────────────────────────────────────────────────

/// Axum handler: serve embedded static assets for the admin UI.
///
/// Maps `GET /admin/<path>` to the embedded asset at `<path>` (defaulting to
/// `index.html` for bare `/admin/`). Returns the appropriate `Content-Type`
/// and `404` for any unknown path.
#[cfg(feature = "admin-ui")]
pub async fn serve_ui(uri: axum::http::Uri) -> impl axum::response::IntoResponse {
    let path = uri.path().trim_start_matches("/admin/");
    let path = if path.is_empty() { "index.html" } else { path };

    match ui_impl::AdminUiAssets::get(path) {
        Some(asset) => axum::response::Response::builder()
            .header("Content-Type", asset.content_type)
            .body(axum::body::Body::from(asset.data))
            .unwrap(),
        None => axum::response::Response::builder()
            .status(404)
            .body(axum::body::Body::from("Not Found"))
            .unwrap(),
    }
}

/// No-op stub when `admin-ui` feature is disabled.
///
/// Any request to `/admin/*` returns `404 Not Found` so the server still
/// compiles cleanly without the feature.
#[cfg(not(feature = "admin-ui"))]
pub async fn serve_ui(_uri: axum::http::Uri) -> impl axum::response::IntoResponse {
    axum::http::StatusCode::NOT_FOUND
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[cfg(feature = "admin-ui")]
    use super::ui_impl::AdminUiAssets;

    /// Verify that all three primary assets are embedded and non-empty.
    #[cfg(feature = "admin-ui")]
    #[test]
    fn test_ui_assets_index_html_embedded() {
        let asset = AdminUiAssets::get("index.html");
        assert!(asset.is_some(), "index.html must be embedded");
        let asset = asset.unwrap();
        assert!(!asset.data.is_empty(), "index.html must not be empty");
        let text = std::str::from_utf8(asset.data).expect("index.html must be valid UTF-8");
        assert!(
            text.contains("Switchboard Admin"),
            "index.html must contain 'Switchboard Admin'"
        );
    }

    #[cfg(feature = "admin-ui")]
    #[test]
    fn test_ui_assets_app_js_embedded() {
        let asset = AdminUiAssets::get("app.js");
        assert!(asset.is_some(), "app.js must be embedded");
        assert!(!asset.unwrap().data.is_empty(), "app.js must not be empty");
    }

    #[cfg(feature = "admin-ui")]
    #[test]
    fn test_ui_assets_style_css_embedded() {
        let asset = AdminUiAssets::get("style.css");
        assert!(asset.is_some(), "style.css must be embedded");
        assert!(
            !asset.unwrap().data.is_empty(),
            "style.css must not be empty"
        );
    }

    #[cfg(feature = "admin-ui")]
    #[test]
    fn test_ui_assets_missing_returns_none() {
        let asset = AdminUiAssets::get("nonexistent.xyz");
        assert!(asset.is_none(), "unknown asset must return None");
    }
}
