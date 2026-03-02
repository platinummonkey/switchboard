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
///
/// `tls_ca_cert_pem` — when `Some`, the server is expected to listen on HTTPS.
/// The readiness probe uses `danger_accept_invalid_certs` to avoid CA-chain
/// setup complexity in the probe loop.
///
/// `tls_client_identity_pem` — when `Some`, the readiness probe presents a
/// client certificate (cert + key concatenated in PEM format). Required when
/// mTLS is enabled, because the server rejects connections without a client
/// certificate at the TLS handshake level.
pub async fn start_test_server(
    config: ServerConfig,
    admin_enabled: bool,
    config_path: String,
    tls_ca_cert_pem: Option<Vec<u8>>,
    tls_client_identity_pem: Option<Vec<u8>>,
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
            &config_path,
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

    wait_for_ready(
        proxy_addr,
        tls_ca_cert_pem.as_deref(),
        tls_client_identity_pem.as_deref(),
    )
    .await;
    (proxy_addr, admin_addr, shutdown_tx)
}

/// Poll `GET {scheme}://{addr}/health` with the test API key until it returns
/// 200 or 5 seconds pass.
///
/// The health endpoint is behind the auth middleware, so the test API key
/// `"test-api-key"` must be included to receive 200 OK rather than 401.
///
/// When `ca_cert_pem` is `Some`, an HTTPS client is used. The probe uses
/// `danger_accept_invalid_certs` to avoid CA-chain bootstrapping complexity.
/// When `client_identity_pem` is also `Some`, the probe presents a client
/// certificate, which is required when the server enforces mTLS.
async fn wait_for_ready(
    addr: SocketAddr,
    ca_cert_pem: Option<&[u8]>,
    client_identity_pem: Option<&[u8]>,
) {
    let (client, url) = if let Some(ca_pem) = ca_cert_pem {
        // HTTPS readiness probe.
        //
        // For mTLS (client_identity_pem is Some): use proper CA verification
        // AND a client identity.  reqwest's rustls backend requires CA trust
        // when identity is also set — `danger_accept_invalid_certs` causes
        // reqwest to use the native-TLS code path, which rejects rustls PEM
        // identities with "incompatible TLS identity type".
        //
        // For one-way TLS (client_identity_pem is None): skip cert
        // verification in the probe loop for simplicity.
        let url = format!("https://{}/health", addr);
        if let Some(identity_pem) = client_identity_pem {
            // mTLS path: full verification + client identity.
            // Must use use_rustls_tls() to select the rustls backend so that
            // reqwest::Identity::from_pem (ClientCert::Pem) is accepted.
            // The default reqwest backend on macOS is native-tls which rejects
            // PEM identities with "incompatible TLS identity type".
            let ca_cert = reqwest::Certificate::from_pem(ca_pem)
                .expect("valid CA cert PEM for mTLS readiness probe");
            let identity = reqwest::Identity::from_pem(identity_pem)
                .expect("valid mTLS identity PEM for readiness probe");
            let client = reqwest::Client::builder()
                .use_rustls_tls()
                .add_root_certificate(ca_cert)
                .identity(identity)
                .timeout(std::time::Duration::from_secs(3))
                .build()
                .expect("failed to build mTLS reqwest client for readiness probe");
            (client, url)
        } else {
            // One-way TLS path: skip cert verification in the probe loop.
            // The actual test client performs full verification.
            let client = reqwest::Client::builder()
                .danger_accept_invalid_certs(true)
                .timeout(std::time::Duration::from_secs(3))
                .build()
                .expect("failed to build TLS reqwest client for readiness probe");
            (client, url)
        }
    } else {
        let client = reqwest::Client::new();
        let url = format!("http://{}/health", addr);
        (client, url)
    };
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);

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
                "test server at {} did not become ready within 10 seconds",
                addr
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}
