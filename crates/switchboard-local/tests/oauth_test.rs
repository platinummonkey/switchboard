//! Integration tests for the OAuth2 client_credentials flow in switchboard-local.
//!
//! Each test spins up a [`wiremock::MockServer`] to simulate the OAuth2 token
//! endpoint and exercises `fetch_oauth_token` / `LocalAuthManager` in oauth mode.

use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use switchboard_local::auth::{LocalAuthManager, fetch_oauth_token};
use switchboard_local::config::{AuthConfig, OAuthConfig};

// ── helpers ───────────────────────────────────────────────────────────────────

fn ok_token_response() -> serde_json::Value {
    serde_json::json!({
        "access_token": "tok-123",
        "expires_in": 3600,
        "token_type": "Bearer"
    })
}

// ── test_oauth_fetch_token ────────────────────────────────────────────────────

/// wiremock returns a valid token response — `fetch_oauth_token` should
/// return `Ok(("tok-123", 3600))`.
#[tokio::test]
async fn test_oauth_fetch_token() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(ok_token_response())
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let client = reqwest::Client::new();
    let token_url = format!("{}/token", mock_server.uri());

    let result = fetch_oauth_token(&client, &token_url, "my-client-id", None).await;
    let (token, expires_in) = result.expect("expected Ok from fetch_oauth_token");

    assert_eq!(token, "tok-123");
    assert_eq!(expires_in, 3600);
}

// ── test_oauth_fetch_token_with_secret ───────────────────────────────────────

/// When `client_secret` is `Some`, the POST body must contain the
/// `client_secret` field.
#[tokio::test]
async fn test_oauth_fetch_token_with_secret() {
    let mock_server = MockServer::start().await;

    // Only match if `client_secret` is present in the body.
    Mock::given(method("POST"))
        .and(path("/token"))
        .and(body_string_contains("client_secret=super-secret"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(ok_token_response())
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let client = reqwest::Client::new();
    let token_url = format!("{}/token", mock_server.uri());

    let result = fetch_oauth_token(&client, &token_url, "my-client-id", Some("super-secret")).await;
    let (token, _) = result.expect("expected Ok when client_secret is provided");
    assert_eq!(token, "tok-123");
}

// ── test_oauth_fetch_token_no_secret ─────────────────────────────────────────

/// When `client_secret` is `None`, the POST body must NOT contain
/// `client_secret`.
#[tokio::test]
async fn test_oauth_fetch_token_no_secret() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(ok_token_response())
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let client = reqwest::Client::new();
    let token_url = format!("{}/token", mock_server.uri());

    let result = fetch_oauth_token(&client, &token_url, "my-client-id", None).await;
    result.expect("fetch_oauth_token should succeed without a client_secret");

    // Inspect the recorded request to confirm `client_secret` is absent.
    let received = mock_server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let body = String::from_utf8(received[0].body.clone()).unwrap();
    assert!(
        !body.contains("client_secret"),
        "body should NOT contain client_secret when None: {body}"
    );
}

// ── test_oauth_fetch_token_server_error ──────────────────────────────────────

/// When the token endpoint returns 500, `fetch_oauth_token` should return an
/// `Err(LocalError::Auth(...))`.
#[tokio::test]
async fn test_oauth_fetch_token_server_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let client = reqwest::Client::new();
    let token_url = format!("{}/token", mock_server.uri());

    let result = fetch_oauth_token(&client, &token_url, "my-client-id", None).await;
    assert!(
        result.is_err(),
        "expected Err when token endpoint returns 500"
    );
    let err = result.unwrap_err();
    let err_str = err.to_string();
    assert!(
        err_str.contains("500"),
        "error message should mention status 500: {err_str}"
    );
}

// ── test_oauth_fetch_token_bad_json ──────────────────────────────────────────

/// When the token endpoint returns `{}` (missing `access_token`), parsing
/// should fail and return an `Err`.
#[tokio::test]
async fn test_oauth_fetch_token_bad_json() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let client = reqwest::Client::new();
    let token_url = format!("{}/token", mock_server.uri());

    let result = fetch_oauth_token(&client, &token_url, "my-client-id", None).await;
    assert!(
        result.is_err(),
        "expected Err when access_token field is missing"
    );
}

// ── test_oauth_get_header_format ──────────────────────────────────────────────

/// `LocalAuthManager` in oauth mode should return an `Authorization` header
/// whose value starts with `"Bearer "`.
#[tokio::test]
async fn test_oauth_get_header_format() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(ok_token_response())
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let token_url = format!("{}/token", mock_server.uri());

    let config = AuthConfig {
        method: "oauth".into(),
        oauth: OAuthConfig {
            client_id: "test-client".into(),
            token_url,
            client_secret: None,
        },
        ..AuthConfig::default()
    };

    let manager = LocalAuthManager::new(&config)
        .await
        .expect("LocalAuthManager::new should succeed with a valid OAuth config");

    let (header_name, header_value) = manager
        .get_header()
        .await
        .expect("get_header should succeed in oauth mode");

    assert_eq!(
        header_name.to_lowercase(),
        "authorization",
        "header name should be 'authorization'"
    );
    assert!(
        header_value.starts_with("Bearer "),
        "header value should start with 'Bearer ': {header_value}"
    );
    assert_eq!(
        header_value, "Bearer tok-123",
        "header value should contain the token from the mock"
    );
}
