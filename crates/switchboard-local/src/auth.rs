//! Local proxy auth token management.
//!
//! [`LocalAuthManager`] reads the auth configuration from [`crate::config::AuthConfig`]
//! and provides an `Authorization` header for outgoing requests to `switchboard-server`.
//!
//! Supported methods:
//! - `api_key` — returns `Authorization: Bearer <key>` from config
//! - `jwt` — uses a static token or runs a shell command to fetch one; background
//!   refresh loop re-runs the command before the token expires
//! - `mtls` — stub (deferred to a later phase)
//! - `oauth` — OAuth2 client_credentials grant with background token refresh

use std::sync::Arc;

use jsonwebtoken::{DecodingKey, Validation, decode};
use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::instrument;

use crate::config::AuthConfig;
use crate::error::LocalError;

// ── JWT claims (minimal, for exp extraction) ──────────────────────────────────

/// Minimal JWT payload — we only need the `exp` claim to schedule refresh.
#[derive(Debug, Deserialize)]
struct JwtExp {
    exp: Option<i64>,
}

// ── Duration parser (mirrors server config/duration.rs) ───────────────────────

/// Parse a human-readable duration string (`"15m"`, `"30s"`, `"2h"`, `"500ms"`)
/// into a [`std::time::Duration`].
fn parse_duration(s: &str) -> Result<std::time::Duration, LocalError> {
    if let Some(v) = s.strip_suffix("ms") {
        let n: u64 = v
            .trim()
            .parse()
            .map_err(|_| LocalError::Auth(format!("invalid refresh_interval '{s}': bad number")))?;
        Ok(std::time::Duration::from_millis(n))
    } else if let Some(v) = s.strip_suffix('s') {
        let n: u64 = v
            .trim()
            .parse()
            .map_err(|_| LocalError::Auth(format!("invalid refresh_interval '{s}': bad number")))?;
        Ok(std::time::Duration::from_secs(n))
    } else if let Some(v) = s.strip_suffix('m') {
        let n: u64 = v
            .trim()
            .parse()
            .map_err(|_| LocalError::Auth(format!("invalid refresh_interval '{s}': bad number")))?;
        Ok(std::time::Duration::from_secs(n * 60))
    } else if let Some(v) = s.strip_suffix('h') {
        let n: u64 = v
            .trim()
            .parse()
            .map_err(|_| LocalError::Auth(format!("invalid refresh_interval '{s}': bad number")))?;
        Ok(std::time::Duration::from_secs(n * 3600))
    } else {
        Err(LocalError::Auth(format!(
            "invalid refresh_interval '{s}': expected suffix ms, s, m, or h"
        )))
    }
}

// ── Token acquisition helpers ─────────────────────────────────────────────────

/// Execute a shell command and return its trimmed stdout as the token.
fn run_token_command(cmd: &str) -> Result<String, LocalError> {
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .map_err(|e| LocalError::Auth(format!("failed to run token_command: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(LocalError::Auth(format!(
            "token_command exited with status {}: {stderr}",
            output.status
        )));
    }

    let token = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if token.is_empty() {
        return Err(LocalError::Auth(
            "token_command produced empty output".into(),
        ));
    }
    Ok(token)
}

/// Extract the `exp` Unix timestamp from a JWT without verifying the signature.
///
/// We intentionally skip signature validation here — we only need the expiry
/// time to schedule a refresh.  The server will perform full validation.
fn extract_exp(token: &str) -> Option<i64> {
    // Use `dangerous_insecure_decode` via validation with all checks disabled.
    let mut validation = Validation::default();
    validation.insecure_disable_signature_validation();
    validation.validate_exp = false;
    validation.validate_aud = false;
    validation.required_spec_claims.clear();

    decode::<JwtExp>(token, &DecodingKey::from_secret(&[]), &validation)
        .ok()
        .and_then(|td| td.claims.exp)
}

// ── Inner state for JWT mode ──────────────────────────────────────────────────

#[derive(Debug)]
struct JwtState {
    /// Current bearer token.
    token: Arc<RwLock<String>>,
    /// Shell command to (re-)fetch the token; `None` for static tokens.
    token_command: Option<String>,
    /// How long before expiry to trigger a refresh.
    #[allow(dead_code)]
    refresh_before_expiry: std::time::Duration,
}

// ── Inner state for OAuth2 mode ───────────────────────────────────────────────

/// OAuth2 token response from the token endpoint.
#[derive(Debug, Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    expires_in: Option<u64>,
}

#[derive(Debug)]
struct OAuthState {
    /// Current access token.
    token: Arc<RwLock<String>>,
    /// The reqwest client shared with the refresh loop.
    client: reqwest::Client,
    /// OAuth2 token endpoint URL.
    token_url: String,
    /// OAuth2 client ID.
    client_id: String,
    /// OAuth2 client secret (optional — omitted from request body if `None`).
    client_secret: Option<String>,
}

// ── OAuth2 token fetch helper ─────────────────────────────────────────────────

/// Fetch an OAuth2 access token via the client_credentials grant.
///
/// POSTs `grant_type=client_credentials&client_id=…[&client_secret=…]` to
/// `token_url` and returns `(access_token, expires_in_secs)`.
pub async fn fetch_oauth_token(
    client: &reqwest::Client,
    token_url: &str,
    client_id: &str,
    client_secret: Option<&str>,
) -> Result<(String, u64), LocalError> {
    let mut params = vec![
        ("grant_type", "client_credentials"),
        ("client_id", client_id),
    ];
    if let Some(secret) = client_secret {
        params.push(("client_secret", secret));
    }

    let response = client
        .post(token_url)
        .header("content-type", "application/x-www-form-urlencoded")
        .form(&params)
        .send()
        .await
        .map_err(|e| LocalError::Auth(format!("OAuth token request failed: {e}")))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<unreadable body>".into());
        return Err(LocalError::Auth(format!(
            "OAuth token endpoint returned {status}: {body}"
        )));
    }

    let token_resp: OAuthTokenResponse = response
        .json()
        .await
        .map_err(|e| LocalError::Auth(format!("failed to parse OAuth token response: {e}")))?;

    let expires_in = token_resp.expires_in.unwrap_or(3600);
    Ok((token_resp.access_token, expires_in))
}

// ── LocalAuthManager ──────────────────────────────────────────────────────────

/// Manages outgoing authentication headers for `switchboard-local`.
///
/// Constructed once at startup via [`LocalAuthManager::new`].  For JWT mode a
/// background refresh loop is started automatically.
#[derive(Debug)]
pub struct LocalAuthManager {
    inner: Inner,
}

#[derive(Debug)]
enum Inner {
    ApiKey(String),
    Jwt(JwtState),
    /// Stub — deferred to a later phase.
    Mtls,
    /// OAuth2 client_credentials grant.
    OAuth(OAuthState),
}

impl LocalAuthManager {
    /// Create a new [`LocalAuthManager`] from the given [`AuthConfig`].
    ///
    /// For JWT mode, a background Tokio task is spawned that refreshes the
    /// token before it expires.
    #[instrument(skip(config), fields(method = %config.method))]
    pub async fn new(config: &AuthConfig) -> Result<Self, LocalError> {
        match config.method.as_str() {
            "api_key" => {
                let key = config.api_key.clone().ok_or_else(|| {
                    LocalError::Auth("auth.method is 'api_key' but auth.api_key is not set".into())
                })?;
                tracing::info!("local auth: api_key mode");
                Ok(Self {
                    inner: Inner::ApiKey(key),
                })
            }

            "jwt" => {
                // Obtain the initial token.
                let initial_token = if let Some(token) = config.jwt.token.clone() {
                    tracing::debug!("local auth: using static JWT token from config");
                    token
                } else if let Some(cmd) = config.jwt.token_command.clone() {
                    tracing::info!(command = %cmd, "local auth: running token_command for initial JWT");
                    run_token_command(&cmd)?
                } else {
                    return Err(LocalError::Auth(
                        "auth.method is 'jwt' but neither auth.jwt.token \
                         nor auth.jwt.token_command is set"
                            .into(),
                    ));
                };

                let refresh_before_expiry = parse_duration(&config.jwt.refresh_interval)?;

                let token_arc = Arc::new(RwLock::new(initial_token.clone()));

                // Start a background refresh loop only when there is a command
                // to re-run.  Static tokens cannot be refreshed automatically.
                if let Some(cmd) = config.jwt.token_command.clone() {
                    let token_arc_bg = Arc::clone(&token_arc);
                    let initial = initial_token.clone();
                    tokio::spawn(async move {
                        jwt_refresh_loop(token_arc_bg, initial, cmd, refresh_before_expiry).await;
                    });
                }

                Ok(Self {
                    inner: Inner::Jwt(JwtState {
                        token: token_arc,
                        token_command: config.jwt.token_command.clone(),
                        refresh_before_expiry,
                    }),
                })
            }

            "mtls" => {
                tracing::warn!("local auth: mTLS mode is not yet implemented in switchboard-local");
                Ok(Self { inner: Inner::Mtls })
            }

            "oauth" => {
                let oauth_cfg = &config.oauth;
                if oauth_cfg.client_id.is_empty() {
                    return Err(LocalError::Auth(
                        "auth.method is 'oauth' but auth.oauth.client_id is not set".into(),
                    ));
                }
                if oauth_cfg.token_url.is_empty() {
                    return Err(LocalError::Auth(
                        "auth.method is 'oauth' but auth.oauth.token_url is not set".into(),
                    ));
                }

                let client = reqwest::Client::new();
                let (initial_token, expires_in) = fetch_oauth_token(
                    &client,
                    &oauth_cfg.token_url,
                    &oauth_cfg.client_id,
                    oauth_cfg.client_secret.as_deref(),
                )
                .await?;

                tracing::info!(
                    client_id = %oauth_cfg.client_id,
                    expires_in,
                    "local auth: OAuth2 client_credentials token acquired"
                );

                let token_arc = Arc::new(RwLock::new(initial_token));

                let oauth_state = OAuthState {
                    token: Arc::clone(&token_arc),
                    client: client.clone(),
                    token_url: oauth_cfg.token_url.clone(),
                    client_id: oauth_cfg.client_id.clone(),
                    client_secret: oauth_cfg.client_secret.clone(),
                };

                // Spawn the background refresh loop.
                {
                    let token_arc_bg = Arc::clone(&token_arc);
                    let token_url = oauth_cfg.token_url.clone();
                    let client_id = oauth_cfg.client_id.clone();
                    let client_secret = oauth_cfg.client_secret.clone();
                    tokio::spawn(async move {
                        oauth_refresh_loop(
                            token_arc_bg,
                            client,
                            token_url,
                            client_id,
                            client_secret,
                            expires_in,
                        )
                        .await;
                    });
                }

                Ok(Self {
                    inner: Inner::OAuth(oauth_state),
                })
            }

            other => Err(LocalError::Auth(format!(
                "unknown auth method '{other}': expected api_key, jwt, mtls, or oauth"
            ))),
        }
    }

    /// Returns the `(header_name, header_value)` pair to inject into outgoing
    /// requests to `switchboard-server`.
    pub async fn get_header(&self) -> Result<(String, String), LocalError> {
        match &self.inner {
            Inner::ApiKey(key) => Ok(("Authorization".into(), format!("Bearer {key}"))),

            Inner::Jwt(state) => {
                let token = state.token.read().await.clone();
                Ok(("Authorization".into(), format!("Bearer {token}")))
            }

            Inner::Mtls => Err(LocalError::Auth(
                "mTLS auth is not yet implemented; cannot produce a header".into(),
            )),

            Inner::OAuth(state) => {
                let token = state.token.read().await.clone();
                Ok(("authorization".into(), format!("Bearer {token}")))
            }
        }
    }

    /// Force a token refresh.  Useful after receiving a `401` from the server.
    ///
    /// - For `api_key`: no-op (static credentials).
    /// - For `jwt` with a `token_command`: re-runs the command and updates the
    ///   stored token.
    /// - For `jwt` with a static token: no-op (nothing to refresh).
    /// - For stubs: returns an error.
    pub async fn refresh(&self) -> Result<(), LocalError> {
        match &self.inner {
            Inner::ApiKey(_) => {
                // Nothing to refresh for static keys.
                Ok(())
            }

            Inner::Jwt(state) => {
                if let Some(cmd) = &state.token_command {
                    tracing::info!(command = %cmd, "local auth: forced JWT refresh");
                    let new_token = run_token_command(cmd)?;
                    let mut w = state.token.write().await;
                    *w = new_token;
                    Ok(())
                } else {
                    // Static JWT — nothing to refresh.
                    Ok(())
                }
            }

            Inner::Mtls => Err(LocalError::Auth(
                "mTLS auth is not yet implemented; cannot refresh".into(),
            )),

            Inner::OAuth(state) => {
                tracing::info!(
                    client_id = %state.client_id,
                    "local auth: forced OAuth2 token refresh"
                );
                let (new_token, _) = fetch_oauth_token(
                    &state.client,
                    &state.token_url,
                    &state.client_id,
                    state.client_secret.as_deref(),
                )
                .await?;
                let mut w = state.token.write().await;
                *w = new_token;
                Ok(())
            }
        }
    }
}

// ── Background JWT refresh loop ────────────────────────────────────────────────

/// Background task: watches the current token's `exp` claim and re-runs
/// `cmd` shortly before expiry to obtain a fresh token.
///
/// If the token has no `exp` claim the loop falls back to refreshing every
/// `refresh_before_expiry` interval.
async fn jwt_refresh_loop(
    token: Arc<RwLock<String>>,
    initial_token: String,
    cmd: String,
    refresh_before_expiry: std::time::Duration,
) {
    let mut current = initial_token;

    loop {
        // Determine how long to sleep before the next refresh.
        let sleep_duration = compute_sleep(&current, refresh_before_expiry);

        tracing::debug!(
            sleep_secs = sleep_duration.as_secs(),
            "JWT refresh loop: sleeping until next refresh"
        );
        tokio::time::sleep(sleep_duration).await;

        // Attempt to refresh the token.
        match run_token_command(&cmd) {
            Ok(new_token) => {
                tracing::info!("JWT refresh loop: token refreshed successfully");
                current = new_token.clone();
                let mut w = token.write().await;
                *w = new_token;
            }
            Err(e) => {
                tracing::error!(error = %e, "JWT refresh loop: failed to refresh token; will retry");
                // Back off briefly before retrying.
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            }
        }
    }
}

// ── Background OAuth2 refresh loop ───────────────────────────────────────────

/// Background task: refreshes the OAuth2 access token before it expires.
///
/// Sleeps for `expires_in - 30` seconds, then re-fetches.  On error, retries
/// with a 30-second backoff up to 5 times before logging and giving up.
async fn oauth_refresh_loop(
    token: Arc<RwLock<String>>,
    client: reqwest::Client,
    token_url: String,
    client_id: String,
    client_secret: Option<String>,
    initial_expires_in: u64,
) {
    const MAX_RETRIES: u32 = 5;
    const RETRY_BACKOFF_SECS: u64 = 30;
    const REFRESH_MARGIN_SECS: u64 = 30;

    let mut expires_in = initial_expires_in;

    loop {
        let sleep_secs = expires_in.saturating_sub(REFRESH_MARGIN_SECS);
        tracing::debug!(
            sleep_secs,
            "OAuth2 refresh loop: sleeping until next refresh"
        );
        tokio::time::sleep(std::time::Duration::from_secs(sleep_secs)).await;

        let mut last_error: Option<LocalError> = None;
        let mut success = false;

        for attempt in 1..=MAX_RETRIES {
            match fetch_oauth_token(&client, &token_url, &client_id, client_secret.as_deref()).await
            {
                Ok((new_token, new_expires_in)) => {
                    tracing::info!(
                        attempt,
                        new_expires_in,
                        "OAuth2 refresh loop: token refreshed successfully"
                    );
                    expires_in = new_expires_in;
                    let mut w = token.write().await;
                    *w = new_token;
                    success = true;
                    break;
                }
                Err(e) => {
                    tracing::warn!(
                        attempt,
                        error = %e,
                        "OAuth2 refresh loop: refresh attempt failed; retrying in {RETRY_BACKOFF_SECS}s"
                    );
                    last_error = Some(e);
                    if attempt < MAX_RETRIES {
                        tokio::time::sleep(std::time::Duration::from_secs(RETRY_BACKOFF_SECS))
                            .await;
                    }
                }
            }
        }

        if !success {
            tracing::error!(
                error = ?last_error,
                "OAuth2 refresh loop: exhausted {MAX_RETRIES} retries; giving up"
            );
            return;
        }
    }
}

/// Compute how long to sleep before the next refresh attempt.
///
/// If `token` carries an `exp` claim, sleep until `exp - refresh_before_expiry`.
/// Otherwise fall back to sleeping for exactly `refresh_before_expiry`.
fn compute_sleep(token: &str, refresh_before_expiry: std::time::Duration) -> std::time::Duration {
    if let Some(exp) = extract_exp(token) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let secs_until_expiry = exp.saturating_sub(now);
        let secs_to_sleep =
            secs_until_expiry.saturating_sub(refresh_before_expiry.as_secs() as i64);

        if secs_to_sleep > 0 {
            std::time::Duration::from_secs(secs_to_sleep as u64)
        } else {
            // Already expired (or about to) — refresh immediately.
            std::time::Duration::ZERO
        }
    } else {
        // No exp claim — fall back to periodic refresh.
        refresh_before_expiry
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AuthConfig, JwtAuthConfig};

    // ── parse_duration ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_duration_ms() {
        assert_eq!(
            parse_duration("500ms").unwrap(),
            std::time::Duration::from_millis(500)
        );
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(
            parse_duration("30s").unwrap(),
            std::time::Duration::from_secs(30)
        );
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(
            parse_duration("15m").unwrap(),
            std::time::Duration::from_secs(900)
        );
    }

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(
            parse_duration("2h").unwrap(),
            std::time::Duration::from_secs(7200)
        );
    }

    #[test]
    fn test_parse_duration_invalid() {
        assert!(parse_duration("abc").is_err());
        assert!(parse_duration("123").is_err());
        assert!(parse_duration("").is_err());
    }

    // ── api_key mode ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_api_key_mode_returns_bearer_header() {
        let config = AuthConfig {
            method: "api_key".into(),
            api_key: Some("sk-test-key".into()),
            ..AuthConfig::default()
        };
        let manager = LocalAuthManager::new(&config).await.unwrap();
        let (name, value) = manager.get_header().await.unwrap();
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer sk-test-key");
    }

    #[tokio::test]
    async fn test_api_key_mode_missing_key_returns_error() {
        let config = AuthConfig {
            method: "api_key".into(),
            api_key: None,
            ..AuthConfig::default()
        };
        assert!(LocalAuthManager::new(&config).await.is_err());
    }

    #[tokio::test]
    async fn test_api_key_refresh_is_noop() {
        let config = AuthConfig {
            method: "api_key".into(),
            api_key: Some("sk-noop".into()),
            ..AuthConfig::default()
        };
        let manager = LocalAuthManager::new(&config).await.unwrap();
        // Refresh should succeed (no-op) for api_key mode.
        assert!(manager.refresh().await.is_ok());
        // Header unchanged after refresh.
        let (_, value) = manager.get_header().await.unwrap();
        assert_eq!(value, "Bearer sk-noop");
    }

    // ── jwt mode — static token ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_jwt_static_token_returns_bearer_header() {
        let config = AuthConfig {
            method: "jwt".into(),
            jwt: JwtAuthConfig {
                token: Some("my.static.token".into()),
                token_command: None,
                refresh_interval: "15m".into(),
            },
            ..AuthConfig::default()
        };
        let manager = LocalAuthManager::new(&config).await.unwrap();
        let (name, value) = manager.get_header().await.unwrap();
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer my.static.token");
    }

    #[tokio::test]
    async fn test_jwt_static_token_refresh_is_noop() {
        let config = AuthConfig {
            method: "jwt".into(),
            jwt: JwtAuthConfig {
                token: Some("my.static.token".into()),
                token_command: None,
                refresh_interval: "15m".into(),
            },
            ..AuthConfig::default()
        };
        let manager = LocalAuthManager::new(&config).await.unwrap();
        assert!(manager.refresh().await.is_ok());
        let (_, value) = manager.get_header().await.unwrap();
        assert_eq!(value, "Bearer my.static.token");
    }

    #[tokio::test]
    async fn test_jwt_missing_both_token_and_command_returns_error() {
        let config = AuthConfig {
            method: "jwt".into(),
            jwt: JwtAuthConfig {
                token: None,
                token_command: None,
                refresh_interval: "15m".into(),
            },
            ..AuthConfig::default()
        };
        assert!(LocalAuthManager::new(&config).await.is_err());
    }

    // ── jwt mode — token_command ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_jwt_token_command_produces_token() {
        let config = AuthConfig {
            method: "jwt".into(),
            jwt: JwtAuthConfig {
                token: None,
                // Use `echo` so the test doesn't need a real JWT issuer.
                token_command: Some("echo 'cmd.produced.token'".into()),
                refresh_interval: "15m".into(),
            },
            ..AuthConfig::default()
        };
        let manager = LocalAuthManager::new(&config).await.unwrap();
        let (_, value) = manager.get_header().await.unwrap();
        assert_eq!(value, "Bearer cmd.produced.token");
    }

    #[tokio::test]
    async fn test_jwt_token_command_refresh_updates_token() {
        let config = AuthConfig {
            method: "jwt".into(),
            jwt: JwtAuthConfig {
                token: None,
                token_command: Some("echo 'refreshed.token'".into()),
                refresh_interval: "15m".into(),
            },
            ..AuthConfig::default()
        };
        let manager = LocalAuthManager::new(&config).await.unwrap();
        manager.refresh().await.unwrap();
        let (_, value) = manager.get_header().await.unwrap();
        assert_eq!(value, "Bearer refreshed.token");
    }

    #[tokio::test]
    async fn test_jwt_token_command_failing_command_returns_error() {
        let config = AuthConfig {
            method: "jwt".into(),
            jwt: JwtAuthConfig {
                token: None,
                token_command: Some("exit 1".into()),
                refresh_interval: "15m".into(),
            },
            ..AuthConfig::default()
        };
        let result = LocalAuthManager::new(&config).await;
        assert!(result.is_err());
    }

    // ── stub modes ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_mtls_mode_get_header_returns_error() {
        let config = AuthConfig {
            method: "mtls".into(),
            ..AuthConfig::default()
        };
        let manager = LocalAuthManager::new(&config).await.unwrap();
        assert!(manager.get_header().await.is_err());
    }

    #[tokio::test]
    async fn test_oauth_mode_missing_client_id_returns_error() {
        // OAuthConfig default has empty client_id and token_url.
        let config = AuthConfig {
            method: "oauth".into(),
            ..AuthConfig::default()
        };
        assert!(LocalAuthManager::new(&config).await.is_err());
    }

    #[tokio::test]
    async fn test_unknown_method_returns_error() {
        let config = AuthConfig {
            method: "kerberos".into(),
            ..AuthConfig::default()
        };
        assert!(LocalAuthManager::new(&config).await.is_err());
    }

    // ── compute_sleep ─────────────────────────────────────────────────────────

    #[test]
    fn test_compute_sleep_no_exp_uses_fallback() {
        // A token without an `exp` claim falls back to the refresh interval.
        let fallback = std::time::Duration::from_secs(900);
        // This is not a real JWT — extract_exp will return None.
        let sleep = compute_sleep("not.a.jwt", fallback);
        assert_eq!(sleep, fallback);
    }

    // ── extract_exp ───────────────────────────────────────────────────────────

    #[test]
    fn test_extract_exp_invalid_token_returns_none() {
        assert!(extract_exp("garbage").is_none());
    }

    #[test]
    fn test_extract_exp_valid_jwt_returns_some() {
        // Build a minimal HS256 JWT with exp claim using jsonwebtoken.
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
        use serde::Serialize;

        #[derive(Serialize)]
        struct Claims {
            sub: String,
            exp: i64,
        }

        let future_exp = 9_999_999_999i64; // year 2286
        let claims = Claims {
            sub: "test".into(),
            exp: future_exp,
        };
        let token = encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(b"secret"),
        )
        .unwrap();

        let extracted = extract_exp(&token);
        assert_eq!(extracted, Some(future_exp));
    }
}
