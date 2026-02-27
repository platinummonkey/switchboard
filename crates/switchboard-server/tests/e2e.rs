//! End-to-end test harness for switchboard-server.
//!
//! Spins up a real TCP server instance backed by wiremock provider mocks.
//! Each test gets its own server on a pre-allocated port from the reserved
//! range 19100–19999.

#[path = "e2e/assertions.rs"]
mod assertions;
#[path = "e2e/client.rs"]
mod client;
#[path = "e2e/config.rs"]
mod config;
#[path = "e2e/harness.rs"]
mod harness;
#[path = "e2e/mocks/mod.rs"]
mod mocks;
#[path = "e2e/port_allocator.rs"]
mod port_allocator;
#[path = "e2e/scenarios/mod.rs"]
mod scenarios;
#[path = "e2e/server.rs"]
mod server;
