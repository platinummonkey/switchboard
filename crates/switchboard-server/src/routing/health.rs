//! Provider-level health checking.
//!
//! [`ProviderHealthChecker`] keeps a per-provider healthy/unhealthy flag and
//! optionally runs periodic background health checks via
//! [`ProviderHealthChecker::spawn_health_checks`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::key_pool::KeyPool;
use crate::providers::ProviderRegistry;

// ── ProviderStatus ────────────────────────────────────────────────────────────

/// Current health status of an upstream provider.
#[derive(Debug, Clone)]
pub struct ProviderStatus {
    pub healthy: bool,
    pub last_checked: Instant,
    pub consecutive_failures: u32,
}

impl Default for ProviderStatus {
    fn default() -> Self {
        Self {
            healthy: true,
            last_checked: Instant::now(),
            consecutive_failures: 0,
        }
    }
}

// ── ProviderHealthChecker ─────────────────────────────────────────────────────

/// Tracks per-provider health and optionally drives background health checks.
///
/// Internally backed by a [`DashMap`] for lock-free concurrent access.
pub struct ProviderHealthChecker {
    /// provider_name → last known health status
    status: DashMap<String, ProviderStatus>,
}

impl std::fmt::Debug for ProviderHealthChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<String> = self.status.iter().map(|e| e.key().clone()).collect();
        f.debug_struct("ProviderHealthChecker")
            .field("tracked_providers", &names)
            .finish()
    }
}

impl Default for ProviderHealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderHealthChecker {
    /// Create a new, empty health checker.
    pub fn new() -> Self {
        Self {
            status: DashMap::new(),
        }
    }

    /// Record the result of a health probe for `provider`.
    ///
    /// If `healthy` is `true` the failure counter is reset.
    /// If `healthy` is `false` the failure counter is incremented.
    pub fn record(&self, provider: &str, healthy: bool) {
        let mut entry = self.status.entry(provider.to_string()).or_default();

        entry.last_checked = Instant::now();
        if healthy {
            entry.healthy = true;
            entry.consecutive_failures = 0;
            tracing::debug!(provider, "health check: provider is healthy");
        } else {
            entry.consecutive_failures += 1;
            entry.healthy = false;
            tracing::warn!(
                provider,
                failures = entry.consecutive_failures,
                "health check: provider is unhealthy"
            );
        }
    }

    /// Returns `true` if the provider is considered healthy.
    ///
    /// Providers that have never been checked are treated as healthy (unknown
    /// is presumed healthy to avoid rejecting traffic on startup).
    pub fn is_healthy(&self, provider: &str) -> bool {
        self.status.get(provider).map(|s| s.healthy).unwrap_or(true) // unknown → optimistic healthy
    }

    /// Spawn background health-check tasks for all providers in the registry.
    ///
    /// For each provider the loop:
    /// 1. Picks the first eligible key from the key pool (if any).
    /// 2. Calls [`crate::routing::UpstreamProvider::health_check`] with that key.
    /// 3. Records the result via [`ProviderHealthChecker::record`].
    ///
    /// The loop runs every `interval`.  The spawned task runs until the process
    /// exits (no graceful shutdown handle is returned — this is intentional for
    /// the background daemon pattern).
    pub fn spawn_health_checks(
        self: Arc<Self>,
        registry: Arc<ProviderRegistry>,
        key_pools: Arc<HashMap<String, Arc<KeyPool>>>,
        interval: Duration,
    ) {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            // The first tick fires immediately; skip it so we don't check on startup
            // before the server is fully initialised.
            ticker.tick().await;

            loop {
                ticker.tick().await;

                for (name, provider) in registry.iter() {
                    // Pick a healthy key from the pool, if available.
                    let Some(pool) = key_pools.get(name) else {
                        tracing::debug!(
                            provider = name,
                            "health check: no key pool found, skipping"
                        );
                        continue;
                    };

                    let ctx = switchboard_common::types::RequestContext::new();
                    let Some(key_arc) = pool.select(&ctx) else {
                        tracing::debug!(
                            provider = name,
                            "health check: no eligible key in pool, skipping"
                        );
                        continue;
                    };
                    // Clone so we release the lock before .await.
                    let key_snapshot = key_arc.read().unwrap().clone();
                    let result = provider.health_check(&key_snapshot).await;
                    self.record(name, result);
                }
            }
        });
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_provider_is_healthy_by_default() {
        let checker = ProviderHealthChecker::new();
        assert!(
            checker.is_healthy("nonexistent-provider"),
            "unknown providers should be treated as healthy"
        );
    }

    #[test]
    fn test_record_unhealthy_marks_provider() {
        let checker = ProviderHealthChecker::new();
        checker.record("anthropic", false);
        assert!(!checker.is_healthy("anthropic"));
    }

    #[test]
    fn test_record_healthy_restores_provider() {
        let checker = ProviderHealthChecker::new();
        // First mark as unhealthy.
        checker.record("openai", false);
        assert!(!checker.is_healthy("openai"));
        // Then restore to healthy.
        checker.record("openai", true);
        assert!(checker.is_healthy("openai"));
    }

    #[test]
    fn test_consecutive_failures_incremented() {
        let checker = ProviderHealthChecker::new();
        checker.record("bedrock", false);
        checker.record("bedrock", false);
        checker.record("bedrock", false);
        let status = checker.status.get("bedrock").unwrap();
        assert_eq!(status.consecutive_failures, 3);
    }

    #[test]
    fn test_consecutive_failures_reset_on_healthy() {
        let checker = ProviderHealthChecker::new();
        checker.record("vertex", false);
        checker.record("vertex", false);
        checker.record("vertex", true);
        let status = checker.status.get("vertex").unwrap();
        assert_eq!(status.consecutive_failures, 0);
        assert!(status.healthy);
    }

    #[test]
    fn test_last_checked_updated() {
        let checker = ProviderHealthChecker::new();
        let before = Instant::now();
        checker.record("ollama", true);
        let status = checker.status.get("ollama").unwrap();
        assert!(status.last_checked >= before);
    }
}
