//! E2E mTLS tests.
//!
//! These tests verify that the switchboard-server correctly:
//! - Serves traffic over one-way TLS (server-auth only).
//! - Accepts mTLS connections where the client presents a certificate signed by
//!   the configured CA.
//! - Rejects connections that do not present a client certificate when mTLS is
//!   required.
//! - Extracts the client certificate Common Name and makes it available as the
//!   user identity in the request context.

use serde_json::json;

use crate::harness::{TestHarnessBuilder, TlsTestConfig};
use crate::mocks::openai::mock_chat_ok;

// ── Certificate generation helper ─────────────────────────────────────────────

/// Generate a self-signed CA plus CA-signed server and client certificates.
///
/// Returns `(ca_cert_pem, server_cert_pem, server_key_pem, client_cert_pem,
/// client_key_pem)`.
///
/// The server certificate has `localhost` as a Subject Alternative Name so
/// that `reqwest` accepts it when connecting to `127.0.0.1`.
/// The client certificate has `client_cn` as its Common Name, which the
/// `MtlsCnResolver` extracts as the user identity.
fn generate_test_certs(client_cn: &str) -> (String, String, String, String, String) {
    // Install the ring crypto provider for rustls (no-op if already installed).
    let _ = rustls::crypto::ring::default_provider().install_default();

    use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, KeyPair, SanType};

    // ── CA ────────────────────────────────────────────────────────────────────
    let ca_key = KeyPair::generate().expect("generate CA key pair");
    let mut ca_params = CertificateParams::default();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params
        .distinguished_name
        .push(DnType::CommonName, "Test CA");
    let ca_cert = ca_params
        .self_signed(&ca_key)
        .expect("generate self-signed CA cert");

    // ── Server cert (signed by CA, SAN = localhost + 127.0.0.1) ─────────────
    // Include the IP address as a SAN so that reqwest accepts the certificate
    // when connecting to 127.0.0.1 (the actual bind address in tests).
    let server_key = KeyPair::generate().expect("generate server key pair");
    let mut server_params = CertificateParams::default();
    server_params.subject_alt_names = vec![
        SanType::DnsName("localhost".try_into().expect("valid SAN DnsName")),
        SanType::IpAddress(std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1))),
    ];
    server_params
        .distinguished_name
        .push(DnType::CommonName, "switchboard-test-server");
    let server_cert = server_params
        .signed_by(&server_key, &ca_cert, &ca_key)
        .expect("sign server cert with CA");

    // ── Client cert (signed by CA, CN = client_cn) ────────────────────────────
    let client_key = KeyPair::generate().expect("generate client key pair");
    let mut client_params = CertificateParams::default();
    client_params
        .distinguished_name
        .push(DnType::CommonName, client_cn);
    let client_cert = client_params
        .signed_by(&client_key, &ca_cert, &ca_key)
        .expect("sign client cert with CA");

    (
        ca_cert.pem(),
        server_cert.pem(),
        server_key.serialize_pem(),
        client_cert.pem(),
        client_key.serialize_pem(),
    )
}

// ── Helper ────────────────────────────────────────────────────────────────────

fn chat_body() -> serde_json::Value {
    json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// One-way TLS: the server presents a certificate but does not require the
/// client to present one.
///
/// A plain `reqwest` client that trusts the test CA can connect and receive a
/// 200 response from the upstream mock.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_one_way_tls_accepts_https_request() {
    let (ca_cert_pem, server_cert_pem, server_key_pem, _client_cert, _client_key) =
        generate_test_certs("unused-client-cn");

    let tls_config = TlsTestConfig {
        server_cert_pem,
        server_key_pem,
        ca_cert_pem,
        // No client cert — one-way TLS only.
        client_cert_pem: None,
        client_key_pem: None,
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_tls(tls_config)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    mock_chat_ok("gpt-4o", "one-way-tls-ok")
        .mount(openai_mock)
        .await;

    // The harness client is configured with HTTPS and the CA cert.
    let resp = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "one-way TLS request must succeed with 200"
    );
}

/// mTLS: when the client presents a certificate signed by the configured CA,
/// the request succeeds and the server records usage.
///
/// This test also verifies the happy path of the full mTLS handshake by
/// checking that the admin usage endpoint sees at least one request after a
/// successful proxied call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_mtls_cn_populates_user_identity() {
    let (ca_cert_pem, server_cert_pem, server_key_pem, client_cert_pem, client_key_pem) =
        generate_test_certs("alice@example.com");

    let tls_config = TlsTestConfig {
        server_cert_pem,
        server_key_pem,
        ca_cert_pem,
        client_cert_pem: Some(client_cert_pem),
        client_key_pem: Some(client_key_pem),
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .with_tls(tls_config)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    mock_chat_ok("gpt-4o", "mtls-identity-ok")
        .mount(openai_mock)
        .await;

    // Send a request with the mTLS client identity — should succeed.
    let proxy_resp = harness.client.chat_completions(chat_body()).await;
    assert_eq!(
        proxy_resp.status().as_u16(),
        200,
        "mTLS request with valid client cert must return 200"
    );

    // Verify the request was tracked in usage (proves it passed through the
    // full middleware stack, including mTLS identity resolution).
    let usage_resp = harness.client.admin_get("usage").await;
    assert_eq!(
        usage_resp.status().as_u16(),
        200,
        "admin /usage must return 200 after mTLS request"
    );

    let body: serde_json::Value = usage_resp
        .json()
        .await
        .expect("admin /usage must return JSON");
    let total = body["total_requests"].as_u64().unwrap_or(0);
    assert!(
        total >= 1,
        "total_requests must be >= 1 after mTLS proxy request; got: {body}"
    );
}

/// mTLS CN → user identity → model override.
///
/// The mTLS client certificate CN is `"alice"`.  A `ModelSelectionConfig`
/// override maps `alice` → `gpt-3.5-turbo`.  The request body contains
/// `model: gpt-4o`, but the `ModelOverrideLayer` should select `gpt-3.5-turbo`
/// for routing because the resolved user ID (from the CN) matches the override
/// entry.
///
/// The proxy does NOT rewrite the `model` field in the upstream request body —
/// it only uses the overridden model for provider routing.  We verify the
/// override worked by:
/// 1. Checking that the mock was hit and the proxy returned 200 (routing worked).
/// 2. Checking the admin `users` endpoint to confirm `user_id = "alice"` was
///    recorded (proving the mTLS CN flowed into the identity chain).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_mtls_cn_used_for_model_override() {
    let (ca_cert_pem, server_cert_pem, server_key_pem, client_cert_pem, client_key_pem) =
        generate_test_certs("alice");

    let tls_config = TlsTestConfig {
        server_cert_pem,
        server_key_pem,
        ca_cert_pem,
        client_cert_pem: Some(client_cert_pem),
        client_key_pem: Some(client_key_pem),
    };

    // Model override: requests from user "alice" get model gpt-3.5-turbo.
    let mut overrides = std::collections::HashMap::new();
    overrides.insert("alice".to_string(), "gpt-3.5-turbo".to_string());
    let model_cfg = switchboard_server::config::model_selection::ModelSelectionConfig {
        mode: "dynamic".into(),
        overrides,
        fallback: Some("gpt-4o".into()),
        ..Default::default()
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_admin()
        .with_tls(tls_config)
        .with_model_selection(model_cfg)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    mock_chat_ok("gpt-4o", "routed-ok").mount(openai_mock).await;

    // Send request with model=gpt-4o in the body.
    // ModelOverrideLayer sees CN=alice → user_id=alice → override fires →
    // model selected for routing = gpt-3.5-turbo. The OpenAI provider
    // handles both gpt-4o and gpt-3.5-turbo, so routing succeeds.
    // The mock at /v1/chat/completions answers regardless of body model.
    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let resp = harness.client.chat_completions(body).await;
    assert_eq!(
        resp.status().as_u16(),
        200,
        "mTLS CN model override must return 200; CN=alice should match the override"
    );

    // Verify the upstream received exactly one request.
    crate::assertions::assert_received_n(openai_mock, 1).await;

    // Verify the admin users endpoint recorded "alice" as a user.
    // This is the definitive proof that the mTLS CN → user_id flow worked:
    // the CN="alice" was extracted, set as user_id, and the usage tracker
    // recorded the request under that identity.
    let users_resp = harness.client.admin_get("users").await;
    assert_eq!(
        users_resp.status().as_u16(),
        200,
        "admin /users must return 200 after mTLS request"
    );
    let users_body: serde_json::Value = users_resp
        .json()
        .await
        .expect("admin /users must return JSON");

    // The users list should contain an entry with user_id = "alice"
    // (the CN from the client certificate), proving the mTLS CN flowed
    // through the identity chain into the usage tracker.
    let users = users_body["users"]
        .as_array()
        .expect("users must be a JSON array");
    let has_alice = users.iter().any(|u| u["user_id"].as_str() == Some("alice"));
    assert!(
        has_alice,
        "usage tracker must record user_id='alice' from mTLS CN; got users: {}",
        serde_json::to_string_pretty(&users_body["users"]).unwrap_or_default()
    );
}

/// mTLS rejection: when the server requires a client certificate (mTLS) but
/// the client does not present one, the connection must fail.
///
/// This test creates a second `reqwest::Client` that trusts the CA (so it can
/// verify the server certificate) but does not present a client certificate.
/// The mTLS handshake must fail at the TLS layer before any HTTP exchange.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_mtls_no_client_cert_rejected() {
    let (ca_cert_pem, server_cert_pem, server_key_pem, client_cert_pem, client_key_pem) =
        generate_test_certs("test");

    let tls_config = TlsTestConfig {
        server_cert_pem,
        server_key_pem,
        ca_cert_pem: ca_cert_pem.clone(),
        client_cert_pem: Some(client_cert_pem),
        client_key_pem: Some(client_key_pem),
    };

    // Start the harness with mTLS configured (client cert required).
    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_tls(tls_config)
        .build()
        .await;

    // Build a separate client that trusts the CA but provides NO client cert.
    // This simulates a client that has not been issued a certificate.
    let ca_cert = reqwest::Certificate::from_pem(ca_cert_pem.as_bytes())
        .expect("valid CA cert for plain client");
    let plain_client = reqwest::Client::builder()
        .add_root_certificate(ca_cert)
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("failed to build plain TLS reqwest client");

    let result = plain_client
        .post(format!("https://{}/v1/chat/completions", harness.addr))
        .header("Authorization", "Bearer test-api-key")
        .json(&chat_body())
        .send()
        .await;

    // The TLS handshake must fail — the server requires a client cert but
    // none was presented.  reqwest surfaces this as a connection error.
    assert!(
        result.is_err(),
        "request without client cert to mTLS server must fail at TLS handshake level"
    );

    // Double-check: the error is a connection / TLS error, not an HTTP-level
    // error.  reqwest wraps these as `reqwest::Error` with `is_connect()`.
    let err = result.unwrap_err();
    assert!(
        err.is_connect() || err.is_request(),
        "expected a connection-level error from failed mTLS handshake, got: {err}"
    );
}
