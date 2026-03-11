//! TLS termination utilities for the switchboard-server.
//!
//! Provides:
//! - [`build_tls_acceptor`] — builds a `tokio_rustls::TlsAcceptor` from PEM files
//! - [`extract_cn_from_tls_stream`] — extracts the Common Name from the peer certificate
//!   after a successful mTLS handshake

use std::sync::Arc;

use rustls::ServerConfig;
use rustls::pki_types::{
    CertificateDer, PrivateKeyDer, PrivatePkcs1KeyDer, PrivatePkcs8KeyDer, PrivateSec1KeyDer,
};
use tokio_rustls::TlsAcceptor;

use crate::config::ServerListenConfig;
use crate::error::ServerError;

/// Decode all PEM blocks from `data`, yielding `(label, DER bytes)` for each.
fn decode_pem_blocks(data: &[u8]) -> Vec<(String, Vec<u8>)> {
    let text = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    let mut results = Vec::new();
    let mut remaining = text;
    while let Some(begin) = remaining.find("-----BEGIN ") {
        let block = &remaining[begin..];
        if let Some(end_line_start) = block.find("\n-----END ") {
            let end_line_end = block[end_line_start + 1..]
                .find('\n')
                .map(|p| end_line_start + 1 + p + 1)
                .unwrap_or(block.len());
            let block_str = &block[..end_line_end];
            if let Ok((label, der)) = pem_rfc7468::decode_vec(block_str.as_bytes()) {
                results.push((label.to_string(), der));
            }
            remaining = &block[end_line_end..];
        } else {
            break;
        }
    }
    results
}

fn parse_certs(pem_data: &[u8]) -> Result<Vec<CertificateDer<'static>>, ServerError> {
    let certs = decode_pem_blocks(pem_data)
        .into_iter()
        .filter(|(label, _)| label == "CERTIFICATE")
        .map(|(_, der)| CertificateDer::from(der))
        .collect::<Vec<_>>();
    if certs.is_empty() {
        return Err(ServerError::Config(
            "no certificates found in PEM data".into(),
        ));
    }
    Ok(certs)
}

fn parse_private_key(pem_data: &[u8]) -> Result<PrivateKeyDer<'static>, ServerError> {
    for (label, der) in decode_pem_blocks(pem_data) {
        match label.as_str() {
            "PRIVATE KEY" => return Ok(PrivatePkcs8KeyDer::from(der).into()),
            "RSA PRIVATE KEY" => return Ok(PrivatePkcs1KeyDer::from(der).into()),
            "EC PRIVATE KEY" => return Ok(PrivateSec1KeyDer::from(der).into()),
            _ => {}
        }
    }
    Err(ServerError::Config(
        "no private key found in tls_key_path".into(),
    ))
}

/// Build a [`TlsAcceptor`] from the PEM files configured in [`ServerListenConfig`].
///
/// Returns `None` if `tls_cert_path` or `tls_key_path` are absent (plain TCP mode).
/// Returns `Some(acceptor)` if TLS is configured, with optional mTLS client verification
/// if `mtls_ca_path` is also set.
pub async fn build_tls_acceptor(
    config: &ServerListenConfig,
) -> Result<Option<TlsAcceptor>, ServerError> {
    let (cert_path, key_path) = match (&config.tls_cert_path, &config.tls_key_path) {
        (Some(c), Some(k)) => (c, k),
        _ => return Ok(None),
    };

    // Read server cert chain.
    let cert_bytes = tokio::fs::read(cert_path)
        .await
        .map_err(|e| ServerError::Config(format!("read tls_cert_path: {e}")))?;
    let cert_chain = parse_certs(&cert_bytes)
        .map_err(|e| ServerError::Config(format!("parse TLS cert: {e}")))?;

    // Read private key.
    let key_bytes = tokio::fs::read(key_path)
        .await
        .map_err(|e| ServerError::Config(format!("read tls_key_path: {e}")))?;
    let private_key: PrivateKeyDer<'static> = parse_private_key(&key_bytes)?;

    // Build rustls ServerConfig.
    let server_config = if let Some(ca_path) = &config.mtls_ca_path {
        // mTLS: require and verify client certificates.
        let ca_bytes = tokio::fs::read(ca_path)
            .await
            .map_err(|e| ServerError::Config(format!("read mtls_ca_path: {e}")))?;
        let ca_certs = parse_certs(&ca_bytes)
            .map_err(|e| ServerError::Config(format!("parse mTLS CA cert: {e}")))?;

        let mut root_store = rustls::RootCertStore::empty();
        for cert in ca_certs {
            root_store
                .add(cert)
                .map_err(|e| ServerError::Config(format!("add CA cert to root store: {e}")))?;
        }

        let client_verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(root_store))
            .build()
            .map_err(|e| ServerError::Config(format!("build mTLS client verifier: {e}")))?;

        ServerConfig::builder()
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(cert_chain, private_key)
            .map_err(|e| ServerError::Config(format!("build TLS ServerConfig (mTLS): {e}")))?
    } else {
        // One-way TLS: no client certificate required.
        ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(cert_chain, private_key)
            .map_err(|e| ServerError::Config(format!("build TLS ServerConfig: {e}")))?
    };

    Ok(Some(TlsAcceptor::from(Arc::new(server_config))))
}

/// Extract the Common Name (CN) from the peer certificate of a TLS stream.
///
/// Returns `None` if no peer certificate is present (one-way TLS or connection
/// without a client certificate).  Returns `Some(cn_string)` with the first CN
/// found in the Subject DN.
pub fn extract_cn_from_tls_stream(
    stream: &tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
) -> Option<String> {
    let (_, server_conn) = stream.get_ref();
    let certs = server_conn.peer_certificates()?;
    let cert_der = certs.first()?;

    use x509_parser::prelude::*;
    let (_, cert) = X509Certificate::from_der(cert_der.as_ref()).ok()?;
    cert.subject()
        .iter_common_name()
        .next()
        .and_then(|cn| cn.as_str().ok())
        .map(|s| s.to_string())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerListenConfig;

    #[tokio::test]
    async fn test_build_tls_acceptor_no_tls_config_returns_none() {
        let config = ServerListenConfig::default();
        let result = build_tls_acceptor(&config).await.unwrap();
        assert!(result.is_none(), "plain TCP config must return None");
    }

    #[tokio::test]
    async fn test_build_tls_acceptor_only_cert_path_returns_none() {
        let config = ServerListenConfig {
            tls_cert_path: Some("/tmp/server.crt".into()),
            tls_key_path: None,
            ..Default::default()
        };
        let result = build_tls_acceptor(&config).await.unwrap();
        assert!(
            result.is_none(),
            "only cert_path without key_path must return None"
        );
    }

    #[tokio::test]
    async fn test_build_tls_acceptor_only_key_path_returns_none() {
        let config = ServerListenConfig {
            tls_cert_path: None,
            tls_key_path: Some("/tmp/server.key".into()),
            ..Default::default()
        };
        let result = build_tls_acceptor(&config).await.unwrap();
        assert!(
            result.is_none(),
            "only key_path without cert_path must return None"
        );
    }

    #[tokio::test]
    async fn test_build_tls_acceptor_missing_cert_file_errors() {
        let config = ServerListenConfig {
            tls_cert_path: Some("/nonexistent/server.crt".into()),
            tls_key_path: Some("/nonexistent/server.key".into()),
            ..Default::default()
        };
        let result = build_tls_acceptor(&config).await;
        assert!(result.is_err(), "missing cert file must produce an error");
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("read tls_cert_path"),
            "error must mention tls_cert_path, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_build_tls_acceptor_missing_key_file_errors() {
        // Write a dummy cert file so the cert read succeeds.
        use std::io::Write as _;
        let mut cert_file = tempfile::NamedTempFile::new().unwrap();
        // A minimal PEM block that will fail cert parsing (not a real cert).
        cert_file
            .write_all(b"-----BEGIN CERTIFICATE-----\nZA==\n-----END CERTIFICATE-----\n")
            .unwrap();

        let config = ServerListenConfig {
            tls_cert_path: Some(cert_file.path().to_str().unwrap().into()),
            tls_key_path: Some("/nonexistent/server.key".into()),
            ..Default::default()
        };
        let result = build_tls_acceptor(&config).await;
        // The cert read may fail (invalid PEM content), or it may succeed and
        // then fail on the key read. Either way, an error is expected.
        assert!(
            result.is_err(),
            "invalid/missing key file must produce an error"
        );
    }

    #[tokio::test]
    async fn test_build_tls_acceptor_with_rcgen_cert() {
        // Install the ring crypto provider required by rustls in test context.
        let _ = rustls::crypto::ring::default_provider().install_default();

        // Generate a self-signed cert + key using rcgen (available in dev-deps).
        use rcgen::generate_simple_self_signed;
        use std::io::Write as _;

        let subject_alt_names = vec!["localhost".into()];
        let cert = generate_simple_self_signed(subject_alt_names).unwrap();
        let cert_pem = cert.cert.pem();
        let key_pem = cert.signing_key.serialize_pem();

        let mut cert_file = tempfile::NamedTempFile::new().unwrap();
        cert_file.write_all(cert_pem.as_bytes()).unwrap();

        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        key_file.write_all(key_pem.as_bytes()).unwrap();

        let config = ServerListenConfig {
            tls_cert_path: Some(cert_file.path().to_str().unwrap().into()),
            tls_key_path: Some(key_file.path().to_str().unwrap().into()),
            mtls_ca_path: None,
            ..Default::default()
        };

        let result = build_tls_acceptor(&config).await;
        assert!(
            result.is_ok(),
            "valid cert+key must succeed: {}",
            result.err().map(|e| e.to_string()).unwrap_or_default()
        );
        assert!(result.unwrap().is_some(), "must return Some(acceptor)");
    }

    #[tokio::test]
    async fn test_build_tls_acceptor_mtls_with_rcgen_certs() {
        // Install the ring crypto provider required by rustls in test context.
        let _ = rustls::crypto::ring::default_provider().install_default();

        use rcgen::generate_simple_self_signed;
        use std::io::Write as _;

        // Server cert.
        let server_cert = generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let server_cert_pem = server_cert.cert.pem();
        let server_key_pem = server_cert.signing_key.serialize_pem();

        // CA cert (reuse another self-signed for simplicity — in real mTLS this
        // would be a proper CA).
        let ca_cert = generate_simple_self_signed(vec!["ca.example.com".into()]).unwrap();
        let ca_pem = ca_cert.cert.pem();

        let mut cert_file = tempfile::NamedTempFile::new().unwrap();
        cert_file.write_all(server_cert_pem.as_bytes()).unwrap();

        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        key_file.write_all(server_key_pem.as_bytes()).unwrap();

        let mut ca_file = tempfile::NamedTempFile::new().unwrap();
        ca_file.write_all(ca_pem.as_bytes()).unwrap();

        let config = ServerListenConfig {
            tls_cert_path: Some(cert_file.path().to_str().unwrap().into()),
            tls_key_path: Some(key_file.path().to_str().unwrap().into()),
            mtls_ca_path: Some(ca_file.path().to_str().unwrap().into()),
            ..Default::default()
        };

        let result = build_tls_acceptor(&config).await;
        assert!(
            result.is_ok(),
            "valid mTLS config must succeed: {}",
            result.err().map(|e| e.to_string()).unwrap_or_default()
        );
        assert!(result.unwrap().is_some(), "mTLS must return Some(acceptor)");
    }

    #[tokio::test]
    async fn test_build_tls_acceptor_mtls_missing_ca_file_errors() {
        use rcgen::generate_simple_self_signed;
        use std::io::Write as _;

        let cert = generate_simple_self_signed(vec!["localhost".into()]).unwrap();

        let mut cert_file = tempfile::NamedTempFile::new().unwrap();
        cert_file.write_all(cert.cert.pem().as_bytes()).unwrap();

        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        key_file
            .write_all(cert.signing_key.serialize_pem().as_bytes())
            .unwrap();

        let config = ServerListenConfig {
            tls_cert_path: Some(cert_file.path().to_str().unwrap().into()),
            tls_key_path: Some(key_file.path().to_str().unwrap().into()),
            mtls_ca_path: Some("/nonexistent/ca.crt".into()),
            ..Default::default()
        };

        let result = build_tls_acceptor(&config).await;
        assert!(result.is_err(), "missing CA file must produce an error");
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("read mtls_ca_path"),
            "error must mention mtls_ca_path, got: {err}"
        );
    }
}
