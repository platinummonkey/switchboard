//! Client→server authentication validators and the upstream `AuthProvider` trait.
//!
//! # Client Auth (incoming requests)
//!
//! Validates that requests from developer tools and CI bots carry valid
//! credentials before they are proxied upstream.
//!
//! - [`validator::ClientAuthValidator`] — the core trait
//! - [`static_key::StaticKeyValidator`] — static pre-shared API keys
//! - [`jwt::JwtValidator`] — RS256/ES256 JWTs validated against a JWKS
//! - [`mtls::MtlsValidator`] — mTLS stub (full impl in axum-TLS phase)
//! - [`registry::AuthRegistry`] — tries validators in order, first success wins
//! - [`validator::ValidatedClient`] — result inserted into request extensions
//!
//! # Upstream Auth (outgoing requests)
//!
//! [`provider::AuthProvider`] supplies credentials injected into upstream
//! LLM provider requests.

pub mod integration_tests;
pub mod jwt;
pub mod mtls;
pub mod provider;
pub mod registry;
pub mod static_key;
pub mod validator;

// Client-facing auth
pub use registry::{AuthRegistry, AuthRegistryBuilder};
pub use static_key::StaticKeyValidator;
pub use validator::{AuthError, ClientAuthValidator, ValidatedClient};

// Upstream auth
pub use provider::{AuthProvider, UpstreamAuthError, UpstreamCredentials};
