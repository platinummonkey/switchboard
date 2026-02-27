//! switchboard-server binary entry point.
//!
//! Loads configuration, initialises OTel tracing, binds listeners, and
//! delegates to [`switchboard_server::run_server`].

use anyhow::Result;
use tokio::signal;

use switchboard_server::config::{self, ServerConfig};
use switchboard_server::observability;

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

    // Extract addresses before config is moved into run_server.
    let listen_addr = server_config.server.listen.clone();
    let admin_enabled = server_config.admin.enabled;
    let admin_listen_addr = server_config.admin.listen.clone();
    let shutdown_timeout = switchboard_server::config::duration::parse(
        &server_config.server.graceful_shutdown_timeout,
    )
    .unwrap_or(std::time::Duration::from_secs(30));

    // Bind listeners up-front so the ports are known before the server starts.
    let proxy_listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    let admin_listener = if admin_enabled {
        Some(tokio::net::TcpListener::bind(&admin_listen_addr).await?)
    } else {
        None
    };

    switchboard_server::run_server(
        server_config,
        &config_path,
        proxy_listener,
        admin_listener,
        shutdown_signal(shutdown_timeout),
    )
    .await
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
