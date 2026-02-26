//! Core proxy request pipeline.
//!
//! # Modules
//!
//! - [`error`] — `ProxyError` type and its `IntoResponse` impl
//! - [`handler`] — axum route handlers and `AppState`
//! - [`stream`] — SSE stream forwarding utilities
//! - [`transform`] — OpenAI ↔ internal ↔ Anthropic format conversions

pub mod error;
pub mod handler;
pub mod stream;
pub mod transform;

pub use error::ProxyError;
pub use handler::AppState;
