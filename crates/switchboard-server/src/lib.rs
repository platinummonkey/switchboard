//! switchboard-server library crate.
//!
//! Re-exports all public modules so they can be referenced from integration
//! tests without duplicating module declarations.

pub mod auth;
pub mod config;
pub mod error;
pub mod guardrails;
pub mod identity;
pub mod key_pool;
pub mod middleware;
pub mod providers;
pub mod proxy;
pub mod routing;
