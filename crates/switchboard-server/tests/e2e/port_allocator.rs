//! Atomic port allocator for E2E tests.
//!
//! Assigns ports from the reserved range 19100–19999. Each port is used at
//! most once per `cargo test` invocation, so TIME_WAIT from completed tests
//! never causes `EADDRINUSE` for later ones.
//!
//! This range sits below both Linux (32768+) and macOS (49152+) ephemeral
//! port ranges, so it does not interfere with OS-assigned ports.

use std::sync::atomic::{AtomicU16, Ordering};

static NEXT_PORT: AtomicU16 = AtomicU16::new(19100);
const PORT_RANGE_END: u16 = 19999;

/// Allocate the next available test port.
///
/// # Panics
///
/// Panics if the port range 19100–19999 is exhausted (would require > 900
/// concurrent E2E server instances, which is not expected in practice).
pub fn allocate() -> u16 {
    let port = NEXT_PORT.fetch_add(1, Ordering::Relaxed);
    assert!(
        port < PORT_RANGE_END,
        "E2E test port range 19100–19999 exhausted; increase PORT_RANGE_END or reduce parallelism"
    );
    port
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_returns_different_ports() {
        let p1 = allocate();
        let p2 = allocate();
        assert_ne!(p1, p2);
        assert!(p1 >= 19100);
        assert!(p2 >= 19100);
        assert!(p1 < PORT_RANGE_END);
        assert!(p2 < PORT_RANGE_END);
    }
}
