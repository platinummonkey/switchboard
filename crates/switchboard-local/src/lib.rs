//! Library entry-point for `switchboard-local`.
//!
//! The binary target (`main.rs`) is the user-facing entry-point.  This
//! `lib.rs` re-exports the internal modules so that integration tests in
//! `tests/` can import them as `switchboard_local::<module>`.

pub mod auth;
pub mod config;
pub mod error;
pub mod model_prefs;
pub mod proxy;
pub mod server;
