//! Server startup for E2E tests.
//!
//! Binds the switchboard server on two ports from the pre-allocated test
//! range (19100–19999): one for the proxy, one (optionally) for admin.
//! Waits for the server to be ready by polling `GET /health`.

use std::net::SocketAddr;

use switchboard_server::config::ServerConfig;

use crate::port_allocator;

/// Start a test server and wait until it responds on `/health`.
///
/// Returns `(proxy_addr, admin_addr, shutdown_tx)`.
/// Drop or send on `shutdown_tx` to stop the server task.
pub async fn start_test_server(
    config: ServerConfig,
    admin_enabled: bool,
) -> (
    SocketAddr,
    Option<SocketAddr>,
    tokio::sync::oneshot::Sender<()>,
) {
    let proxy_port = port_allocator::allocate();
    let proxy_listener = tokio::net::TcpListener::bind(("127.0.0.1", proxy_port))
        .await
        .unwrap_or_else(|e| panic!("failed to bind proxy port {}: {}", proxy_port, e));
    let proxy_addr = proxy_listener.local_addr().unwrap();

    let (admin_listener, admin_addr) = if admin_enabled {
        let admin_port = port_allocator::allocate();
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", admin_port))
            .await
            .unwrap_or_else(|e| panic!("failed to bind admin port {}: {}", admin_port, e));
        let addr = listener.local_addr().unwrap();
        (Some(listener), Some(addr))
    } else {
        (None, None)
    };

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let server_handle = tokio::spawn(async move {
        switchboard_server::run_server(
            config,
            "", // config_path: no file-based hot-reload in tests
            proxy_listener,
            admin_listener,
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
        .unwrap_or_else(|e| {
            // Server exiting due to shutdown_rx being dropped is normal.
            // Log errors that occur before the shutdown signal is sent.
            tracing::warn!(error = %e, "test server exited with error");
        });
    });
    // Yield once so the task has a chance to start.  If it exits immediately
    // (e.g. due to a listen error) we surface a clearer panic than the
    // five-second timeout in wait_for_ready.
    tokio::task::yield_now().await;
    if server_handle.is_finished() {
        panic!("test server task exited immediately after spawning — check for startup errors");
    }

    wait_for_ready(proxy_addr).await;
    (proxy_addr, admin_addr, shutdown_tx)
}

/// Poll `GET http://{addr}/health` with the test API key until it returns 200
/// or 5 seconds pass.
///
/// The health endpoint is behind the auth middleware, so the test API key
/// `"test-api-key"` must be included to receive 200 OK rather than 401.
async fn wait_for_ready(addr: SocketAddr) {
    let client = reqwest::Client::new();
    let url = format!("http://{}/health", addr);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

    loop {
        match client
            .get(&url)
            .header("Authorization", "Bearer test-api-key")
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => return,
            _ => {}
        }
        if tokio::time::Instant::now() >= deadline {
            panic!(
                "test server at {} did not become ready within 5 seconds",
                addr
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}
