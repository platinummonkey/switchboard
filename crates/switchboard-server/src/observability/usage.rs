//! In-memory usage tracking for the switchboard proxy.
//!
//! [`UsageTracker`] accumulates per-user, per-model, and per-provider counters
//! using lock-free [`AtomicU64`] counters inside a [`DashMap`].  It is
//! designed to be shared between the proxy handlers (writers) and the admin API
//! (readers) via `Arc<UsageTracker>`.
//!
//! # Design decisions
//!
//! - `DashMap` provides concurrent map access without a global lock.
//! - `AtomicU64` with `Relaxed` ordering is sufficient for counter increments;
//!   we only need eventual consistency across threads, not strict happens-before.
//! - Snapshots (`totals`, `by_user`) iterate the maps under their shard locks
//!   and produce plain `Vec`/`HashMap` values — no Arc cloning or long-held
//!   locks in hot paths.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

// ── Per-dimension counters ────────────────────────────────────────────────────

/// Per-dimension (user / model / provider) usage counters.
#[derive(Debug, Default)]
pub struct UserCounters {
    pub requests: AtomicU64,
    pub input_tokens: AtomicU64,
    pub output_tokens: AtomicU64,
}

// ── UsageTracker ──────────────────────────────────────────────────────────────

/// Global in-memory usage tracker shared between the proxy handler and admin API.
///
/// All methods are safe to call concurrently from multiple tokio tasks without
/// external synchronisation.
#[derive(Debug, Default)]
pub struct UsageTracker {
    /// Counters keyed by user identifier.
    by_user: DashMap<String, Arc<UserCounters>>,
    /// Counters keyed by model name.
    by_model: DashMap<String, Arc<UserCounters>>,
    /// Counters keyed by provider name.
    by_provider: DashMap<String, Arc<UserCounters>>,
    /// Global totals.
    total_requests: AtomicU64,
    total_input_tokens: AtomicU64,
    total_output_tokens: AtomicU64,
}

impl UsageTracker {
    /// Create a new empty [`UsageTracker`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed (non-streaming) request.
    ///
    /// Increments counters for the given `user_id`, `model`, and `provider` as
    /// well as the global totals.  All updates use `Relaxed` ordering since we
    /// only need eventual consistency.
    pub fn record(
        &self,
        user_id: &str,
        model: &str,
        provider: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) {
        let input = input_tokens as u64;
        let output = output_tokens as u64;

        // Global totals.
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_input_tokens.fetch_add(input, Ordering::Relaxed);
        self.total_output_tokens
            .fetch_add(output, Ordering::Relaxed);

        // Per-user.
        self.increment_counters(&self.by_user, user_id, input, output);
        // Per-model.
        self.increment_counters(&self.by_model, model, input, output);
        // Per-provider.
        self.increment_counters(&self.by_provider, provider, input, output);
    }

    /// Increment the counters for `key` in `map`, creating an entry if absent.
    fn increment_counters(
        &self,
        map: &DashMap<String, Arc<UserCounters>>,
        key: &str,
        input: u64,
        output: u64,
    ) {
        // Fast path: entry already exists.
        if let Some(entry) = map.get(key) {
            entry.requests.fetch_add(1, Ordering::Relaxed);
            entry.input_tokens.fetch_add(input, Ordering::Relaxed);
            entry.output_tokens.fetch_add(output, Ordering::Relaxed);
            return;
        }

        // Slow path: insert a new entry, then increment.  We use
        // `entry().or_insert_with` to avoid a TOCTOU race.
        let counters = map
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(UserCounters::default()));
        counters.requests.fetch_add(1, Ordering::Relaxed);
        counters.input_tokens.fetch_add(input, Ordering::Relaxed);
        counters.output_tokens.fetch_add(output, Ordering::Relaxed);
    }

    /// Snapshot the current global totals plus per-dimension breakdowns.
    pub fn totals(&self) -> UsageTotals {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_input_tokens = self.total_input_tokens.load(Ordering::Relaxed);
        let total_output_tokens = self.total_output_tokens.load(Ordering::Relaxed);

        let requests_by_provider: HashMap<String, u64> = self
            .by_provider
            .iter()
            .map(|e| (e.key().clone(), e.value().requests.load(Ordering::Relaxed)))
            .collect();

        let requests_by_model: HashMap<String, u64> = self
            .by_model
            .iter()
            .map(|e| (e.key().clone(), e.value().requests.load(Ordering::Relaxed)))
            .collect();

        let requests_by_user: HashMap<String, u64> = self
            .by_user
            .iter()
            .map(|e| (e.key().clone(), e.value().requests.load(Ordering::Relaxed)))
            .collect();

        UsageTotals {
            total_requests,
            total_input_tokens,
            total_output_tokens,
            requests_by_provider,
            requests_by_model,
            requests_by_user,
        }
    }

    /// Snapshot per-user stats, sorted by total requests descending.
    pub fn by_user(&self) -> Vec<UserUsage> {
        let mut users: Vec<UserUsage> = self
            .by_user
            .iter()
            .map(|e| UserUsage {
                user_id: e.key().clone(),
                requests: e.value().requests.load(Ordering::Relaxed),
                input_tokens: e.value().input_tokens.load(Ordering::Relaxed),
                output_tokens: e.value().output_tokens.load(Ordering::Relaxed),
            })
            .collect();

        users.sort_by(|a, b| b.requests.cmp(&a.requests));
        users
    }
}

// ── Snapshot types ────────────────────────────────────────────────────────────

/// A point-in-time snapshot of global usage counters.
#[derive(Debug, serde::Serialize)]
pub struct UsageTotals {
    pub total_requests: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub requests_by_provider: HashMap<String, u64>,
    pub requests_by_model: HashMap<String, u64>,
    pub requests_by_user: HashMap<String, u64>,
}

/// A point-in-time snapshot of one user's usage counters.
#[derive(Debug, serde::Serialize)]
pub struct UserUsage {
    pub user_id: String,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_record_increments_total() {
        let tracker = UsageTracker::new();
        tracker.record("alice", "gpt-4o", "openai", 10, 20);
        let totals = tracker.totals();
        assert_eq!(totals.total_requests, 1);
        assert_eq!(totals.total_input_tokens, 10);
        assert_eq!(totals.total_output_tokens, 20);
    }

    #[test]
    fn test_usage_record_by_user() {
        let tracker = UsageTracker::new();
        tracker.record("alice", "gpt-4o", "openai", 5, 10);
        let users = tracker.by_user();
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].user_id, "alice");
        assert_eq!(users[0].requests, 1);
        assert_eq!(users[0].input_tokens, 5);
        assert_eq!(users[0].output_tokens, 10);
    }

    #[test]
    fn test_usage_record_by_model() {
        let tracker = UsageTracker::new();
        tracker.record("bob", "gpt-4o", "openai", 3, 7);
        let totals = tracker.totals();
        assert_eq!(totals.requests_by_model.get("gpt-4o"), Some(&1u64));
    }

    #[test]
    fn test_usage_record_accumulates() {
        let tracker = UsageTracker::new();
        tracker.record("alice", "gpt-4o", "openai", 1, 2);
        tracker.record("alice", "gpt-4o", "openai", 3, 4);
        tracker.record("bob", "gpt-4o-mini", "openai", 5, 6);
        let totals = tracker.totals();
        assert_eq!(totals.total_requests, 3);
        assert_eq!(totals.total_input_tokens, 9);
        assert_eq!(totals.total_output_tokens, 12);
    }

    #[test]
    fn test_usage_totals_includes_all_dimensions() {
        let tracker = UsageTracker::new();
        tracker.record("alice", "gpt-4o", "openai", 10, 20);
        tracker.record("bob", "claude-3", "anthropic", 5, 15);
        let totals = tracker.totals();

        // Provider breakdowns.
        assert_eq!(totals.requests_by_provider.get("openai"), Some(&1u64));
        assert_eq!(totals.requests_by_provider.get("anthropic"), Some(&1u64));

        // Model breakdowns.
        assert_eq!(totals.requests_by_model.get("gpt-4o"), Some(&1u64));
        assert_eq!(totals.requests_by_model.get("claude-3"), Some(&1u64));

        // User breakdowns.
        assert_eq!(totals.requests_by_user.get("alice"), Some(&1u64));
        assert_eq!(totals.requests_by_user.get("bob"), Some(&1u64));
    }

    #[tokio::test]
    async fn test_usage_concurrent_updates() {
        let tracker = Arc::new(UsageTracker::new());
        let mut handles = Vec::new();

        for _ in 0..10 {
            let t = Arc::clone(&tracker);
            handles.push(tokio::spawn(async move {
                for _ in 0..100 {
                    t.record("user", "model", "provider", 1, 1);
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let totals = tracker.totals();
        assert_eq!(totals.total_requests, 1000);
        assert_eq!(totals.total_input_tokens, 1000);
        assert_eq!(totals.total_output_tokens, 1000);
    }
}
