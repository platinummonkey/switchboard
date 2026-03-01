//! Phase 16b: Multi-provider failover and key rotation tests.
//!
//! Tests cover:
//! - Key pool health status transitions (Healthy → Degraded, RateLimited)
//! - WeightedRandom selector never picks disabled or rate-limited keys
//! - RoundRobin selector distributes evenly across healthy keys
//! - Sticky selector routes the same user to the same key
//! - ProviderRegistry resolves providers by config models list
//! - ProviderRegistry falls back to supports_model when config has no models

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use http::{HeaderName, HeaderValue};

use switchboard_server::auth::UpstreamCredentials;
use switchboard_server::config::provider::{KeyPoolConfig, ProviderConfig};
use switchboard_server::key_pool::health::KeyStatus;
use switchboard_server::key_pool::pool::{KeyPool, KeySelector};
use switchboard_server::key_pool::provider::PooledKey;
use switchboard_server::key_pool::selector::{
    RoundRobinSelector, StickySelector, WeightedRandomSelector,
};
use switchboard_server::providers::{AnthropicProvider, OpenAiProvider, ProviderRegistry};

use switchboard_common::types::RequestContext;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_key(id: &str) -> PooledKey {
    PooledKey::new_static(
        id,
        UpstreamCredentials {
            header_name: HeaderName::from_static("authorization"),
            header_value: HeaderValue::from_static("Bearer sk-test"),
            expires_at: None,
        },
        1.0,
    )
}

fn make_key_with_status(id: &str, status: KeyStatus) -> PooledKey {
    let mut k = make_key(id);
    k.health.status = status;
    k
}

fn make_pool_weighted(keys: Vec<PooledKey>) -> KeyPool {
    let arcs: Vec<std::sync::Arc<std::sync::RwLock<PooledKey>>> = keys
        .into_iter()
        .map(|k| std::sync::Arc::new(std::sync::RwLock::new(k)))
        .collect();
    KeyPool::new(arcs, Box::new(WeightedRandomSelector))
}

fn make_pool_round_robin(keys: Vec<PooledKey>) -> KeyPool {
    let arcs: Vec<std::sync::Arc<std::sync::RwLock<PooledKey>>> = keys
        .into_iter()
        .map(|k| std::sync::Arc::new(std::sync::RwLock::new(k)))
        .collect();
    KeyPool::new(arcs, Box::new(RoundRobinSelector::new()))
}

fn make_provider_config(models: Vec<String>, api_format: &str) -> ProviderConfig {
    ProviderConfig {
        base_url: Some("https://example.com".into()),
        api_format: api_format.to_string(),
        models,
        region: None,
        cross_region_inference: false,
        project_id: None,
        timeout: "30s".into(),
        health_check_interval: "30s".into(),
        max_concurrent: 10,
        key_pool: KeyPoolConfig::default(),
    }
}

/// Minimal selector for tests that always picks the first eligible key.
struct FirstEligibleSelector;

impl KeySelector for FirstEligibleSelector {
    fn select(
        &self,
        pool: &[std::sync::Arc<std::sync::RwLock<PooledKey>>],
        _req: &RequestContext,
    ) -> Option<std::sync::Arc<std::sync::RwLock<PooledKey>>> {
        pool.iter()
            .find(|k| k.read().unwrap().is_eligible())
            .map(std::sync::Arc::clone)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// A pool with one `RateLimited` key and one `Healthy` key must never
/// select the rate-limited key across 100 draws.
#[test]
fn test_key_pool_rotates_away_from_rate_limited_key() {
    let rate_limited_key = make_key_with_status("rate-limited", KeyStatus::RateLimited);
    let healthy_key = make_key("healthy");

    let pool = make_pool_weighted(vec![rate_limited_key, healthy_key]);
    let ctx = RequestContext::default();

    for i in 0..100 {
        let selected = pool.select(&ctx).expect("pool must return a key");
        assert_eq!(
            selected.read().unwrap().id,
            "healthy",
            "iteration {i}: rate-limited key must never be selected"
        );
    }
}

/// `KeyPool::record_error` transitions a key to `Degraded` after exceeding the
/// error threshold (5 non-rate-limit errors).  After clearing the counter and
/// calling `recheck_health`, the key returns to `Healthy`.
#[test]
fn test_key_pool_health_transitions() {
    let pool = KeyPool::new(
        vec![std::sync::Arc::new(std::sync::RwLock::new(make_key("k1")))],
        Box::new(FirstEligibleSelector),
    );

    // The key starts healthy.
    {
        let key_arc = pool.get_key_mut("k1").unwrap();
        assert_eq!(key_arc.read().unwrap().health.status, KeyStatus::Healthy);
    }

    // Record 6 non-rate-limit errors.  The threshold is 5, so the 6th error
    // (errors_last_5m goes from 5 to 6, which is > 5) triggers Degraded.
    for _ in 0..6 {
        pool.record_error("k1", false);
    }
    {
        let key_arc = pool.get_key_mut("k1").unwrap();
        assert_eq!(
            key_arc.read().unwrap().health.status,
            KeyStatus::Degraded,
            "key should be Degraded after exceeding error threshold"
        );
    }

    // Simulate a recovery: clear the error counter (as the sliding window would
    // do after 5 minutes) and call recheck_health.
    {
        let key_arc = pool.get_key_mut("k1").unwrap();
        key_arc.write().unwrap().health.errors_last_5m = 0;
    }
    pool.recheck_health();

    {
        let key_arc = pool.get_key_mut("k1").unwrap();
        assert_eq!(
            key_arc.read().unwrap().health.status,
            KeyStatus::Healthy,
            "key should recover to Healthy after errors are cleared"
        );
    }
}

/// `ProviderRegistry::resolve_provider` returns the provider that lists the
/// requested model in its config `models` list, not just any provider that
/// claims to support the model.
///
/// Note: Because `HashMap` iteration order is non-deterministic, this test
/// focuses on the case where only one provider lists the model in its config.
#[test]
fn test_multi_provider_failover_to_backup() {
    // Build a registry with two providers.
    let openai = Arc::new(OpenAiProvider::new_named(
        "openai",
        "https://api.openai.com",
        vec!["gpt-4o".into()],
        Duration::from_secs(10),
    ));
    let anthropic = Arc::new(AnthropicProvider::new_with_base_url(
        "https://api.anthropic.com",
        vec!["claude-sonnet-4-20250514".into()],
        Duration::from_secs(10),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("openai", openai);
    registry.register("anthropic", anthropic);

    let mut config = HashMap::new();
    config.insert(
        "openai".into(),
        make_provider_config(vec!["gpt-4o".into()], "openai"),
    );
    config.insert(
        "anthropic".into(),
        make_provider_config(vec!["claude-sonnet-4-20250514".into()], "anthropic"),
    );

    // gpt-4o resolves to openai.
    let result_openai = registry.resolve_provider("gpt-4o", &config);
    assert!(result_openai.is_some(), "gpt-4o must resolve to a provider");
    let (prov, _) = result_openai.unwrap();
    assert_eq!(prov.name(), "openai");

    // claude resolves to anthropic.
    let result_anthropic = registry.resolve_provider("claude-sonnet-4-20250514", &config);
    assert!(
        result_anthropic.is_some(),
        "claude-sonnet-4-20250514 must resolve to a provider"
    );
    let (prov2, _) = result_anthropic.unwrap();
    assert_eq!(prov2.name(), "anthropic");

    // A model not listed in any config and not supported by any provider
    // returns None.
    assert!(
        registry
            .resolve_provider("unknown-model-xyz", &config)
            .is_none(),
        "unknown model must not resolve"
    );
}

/// When both OpenAI and Anthropic are registered and config lists `gpt-4o`
/// under `openai` and `claude-sonnet-4-20250514` under `anthropic`, the
/// registry must return the correct provider for each model.
#[test]
fn test_provider_registry_resolves_by_config_first() {
    let openai = Arc::new(OpenAiProvider::new_named(
        "openai",
        "https://api.openai.com",
        vec!["gpt-4o".into()],
        Duration::from_secs(10),
    ));
    let anthropic = Arc::new(AnthropicProvider::new_with_base_url(
        "https://api.anthropic.com",
        vec!["claude-sonnet-4-20250514".into()],
        Duration::from_secs(10),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("openai", openai);
    registry.register("anthropic", anthropic);

    let mut config = HashMap::new();
    config.insert(
        "openai".into(),
        make_provider_config(vec!["gpt-4o".into()], "openai"),
    );
    config.insert(
        "anthropic".into(),
        make_provider_config(vec!["claude-sonnet-4-20250514".into()], "anthropic"),
    );

    // gpt-4o must resolve to OpenAI.
    let (prov, model) = registry
        .resolve_provider("gpt-4o", &config)
        .expect("gpt-4o must resolve");
    assert_eq!(prov.name(), "openai", "gpt-4o must go to OpenAI provider");
    assert_eq!(model, "gpt-4o");

    // claude must resolve to Anthropic.
    let (prov2, model2) = registry
        .resolve_provider("claude-sonnet-4-20250514", &config)
        .expect("claude must resolve");
    assert_eq!(
        prov2.name(),
        "anthropic",
        "claude-sonnet-4-20250514 must go to Anthropic provider"
    );
    assert_eq!(model2, "claude-sonnet-4-20250514");
}

/// When config has an empty `models` list for a provider, the registry must
/// fall back to `supports_model()` and still find the provider.
#[test]
fn test_provider_registry_supports_model_fallback() {
    let openai = Arc::new(OpenAiProvider::new_named(
        "openai",
        "https://api.openai.com",
        vec!["gpt-4o".into(), "gpt-4o-mini".into()],
        Duration::from_secs(10),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("openai", openai);

    // Config has no models listed for openai.
    let mut config = HashMap::new();
    config.insert("openai".into(), make_provider_config(vec![], "openai"));

    // Should fall back to supports_model() on the OpenAiProvider.
    let result = registry.resolve_provider("gpt-4o", &config);
    assert!(
        result.is_some(),
        "gpt-4o should be found via supports_model fallback"
    );

    let (prov, _) = result.unwrap();
    assert_eq!(prov.name(), "openai");
    assert_eq!(prov.api_format(), "openai");
}

/// `WeightedRandomSelector` must never select a `Disabled` key regardless of
/// pool composition (1 disabled + 2 healthy keys, 1000 draws).
#[test]
fn test_weighted_random_selector_never_picks_disabled() {
    let disabled_key = make_key_with_status("disabled", KeyStatus::Disabled);
    let healthy_1 = make_key("healthy-1");
    let healthy_2 = make_key("healthy-2");

    let pool_keys: Vec<std::sync::Arc<std::sync::RwLock<PooledKey>>> = vec![
        std::sync::Arc::new(std::sync::RwLock::new(disabled_key)),
        std::sync::Arc::new(std::sync::RwLock::new(healthy_1)),
        std::sync::Arc::new(std::sync::RwLock::new(healthy_2)),
    ];
    let ctx = RequestContext::default();

    for i in 0..1000 {
        let selected = WeightedRandomSelector
            .select(&pool_keys, &ctx)
            .expect("pool must return a key");
        assert_ne!(
            selected.read().unwrap().id,
            "disabled",
            "iteration {i}: disabled key must never be selected"
        );
    }
}

/// `RoundRobinSelector` distributes selections evenly across 3 healthy keys
/// (300 selections → each key selected ~100 times, within 20% tolerance).
#[test]
fn test_round_robin_selector_distributes_evenly() {
    let pool_keys: Vec<std::sync::Arc<std::sync::RwLock<PooledKey>>> = vec![
        std::sync::Arc::new(std::sync::RwLock::new(make_key("k1"))),
        std::sync::Arc::new(std::sync::RwLock::new(make_key("k2"))),
        std::sync::Arc::new(std::sync::RwLock::new(make_key("k3"))),
    ];
    let selector = RoundRobinSelector::new();
    let ctx = RequestContext::default();

    let mut counts: HashMap<String, usize> = HashMap::new();
    counts.insert("k1".into(), 0);
    counts.insert("k2".into(), 0);
    counts.insert("k3".into(), 0);

    for _ in 0..300 {
        let selected = selector
            .select(&pool_keys, &ctx)
            .expect("selector must return a key");
        let id = selected.read().unwrap().id.clone();
        *counts.get_mut(&id).unwrap() += 1;
    }

    // Each key should be selected ~100 times. Allow 20% tolerance (80–120).
    let tolerance = 20usize;
    let expected = 100usize;
    for (id, count) in &counts {
        let diff = if *count >= expected {
            count - expected
        } else {
            expected - count
        };
        assert!(
            diff <= tolerance,
            "key {id} was selected {count} times (expected ~{expected}, tolerance ±{tolerance})"
        );
    }
}

/// `StickySelector` must return the same key for the same `user_id` across
/// multiple calls, regardless of which key the inner selector would have
/// chosen.
#[test]
fn test_sticky_selector_same_user_gets_same_key() {
    let pool_keys: Vec<std::sync::Arc<std::sync::RwLock<PooledKey>>> = vec![
        std::sync::Arc::new(std::sync::RwLock::new(make_key("k1"))),
        std::sync::Arc::new(std::sync::RwLock::new(make_key("k2"))),
        std::sync::Arc::new(std::sync::RwLock::new(make_key("k3"))),
    ];

    // Use a round-robin inner selector so the first call picks k1.
    let inner = Box::new(RoundRobinSelector::new());
    let selector = StickySelector::new(inner);

    let ctx = RequestContext {
        user_id: Some("alice".to_string()),
        ..Default::default()
    };

    // First call determines which key alice is stuck to.
    let first_id = selector
        .select(&pool_keys, &ctx)
        .expect("must select a key")
        .read()
        .unwrap()
        .id
        .clone();

    // Subsequent calls for the same user must always return the same key.
    for i in 0..50 {
        let id = selector
            .select(&pool_keys, &ctx)
            .expect("must select a key")
            .read()
            .unwrap()
            .id
            .clone();
        assert_eq!(
            id, first_id,
            "iteration {i}: alice must always get key '{first_id}'"
        );
    }
}

/// `StickySelector` routes different users to different keys (round-robin
/// assigns k1 → alice, k2 → bob, k3 → carol).
#[test]
fn test_sticky_selector_different_users_get_different_keys() {
    let pool_keys: Vec<std::sync::Arc<std::sync::RwLock<PooledKey>>> = vec![
        std::sync::Arc::new(std::sync::RwLock::new(make_key("k1"))),
        std::sync::Arc::new(std::sync::RwLock::new(make_key("k2"))),
        std::sync::Arc::new(std::sync::RwLock::new(make_key("k3"))),
    ];
    let inner = Box::new(RoundRobinSelector::new());
    let selector = StickySelector::new(inner);

    let alice_ctx = RequestContext {
        user_id: Some("alice".to_string()),
        ..Default::default()
    };
    let bob_ctx = RequestContext {
        user_id: Some("bob".to_string()),
        ..Default::default()
    };
    let carol_ctx = RequestContext {
        user_id: Some("carol".to_string()),
        ..Default::default()
    };

    let alice_key = selector
        .select(&pool_keys, &alice_ctx)
        .unwrap()
        .read()
        .unwrap()
        .id
        .clone();
    let bob_key = selector
        .select(&pool_keys, &bob_ctx)
        .unwrap()
        .read()
        .unwrap()
        .id
        .clone();
    let carol_key = selector
        .select(&pool_keys, &carol_ctx)
        .unwrap()
        .read()
        .unwrap()
        .id
        .clone();

    assert_eq!(alice_key, "k1");
    assert_eq!(bob_key, "k2");
    assert_eq!(carol_key, "k3");
}

/// A `RateLimited` key behaves differently from a `Disabled` key:
/// `KeyStatus::is_eligible()` returns true for `RateLimited` but false for
/// `Disabled`.  The `WeightedRandomSelector` excludes both via effective_weight
/// (RateLimited → 0.0 weight), while `RoundRobinSelector` still considers
/// `RateLimited` as eligible.
#[test]
fn test_key_status_eligibility_semantics() {
    let rate_limited = make_key_with_status("rl", KeyStatus::RateLimited);
    let disabled = make_key_with_status("dis", KeyStatus::Disabled);
    let healthy = make_key("healthy");

    // Eligibility: RateLimited is eligible, Disabled is not.
    assert!(
        rate_limited.is_eligible(),
        "RateLimited keys are technically eligible"
    );
    assert!(!disabled.is_eligible(), "Disabled keys are not eligible");
    assert!(healthy.is_eligible(), "Healthy keys are eligible");

    // Effective weight: both RateLimited and Disabled produce 0.0.
    assert!(
        rate_limited.effective_weight() < f64::EPSILON,
        "RateLimited effective weight must be 0"
    );
    assert!(
        disabled.effective_weight() < f64::EPSILON,
        "Disabled effective weight must be 0"
    );
    assert!(
        healthy.effective_weight() > 0.0,
        "Healthy effective weight must be positive"
    );
}

/// `KeyPool::record_rate_limit` is a convenience wrapper that sets a key to
/// `RateLimited` status.  After `recheck_health` with the rate-limit counter
/// still positive, the key stays `RateLimited`.
#[test]
fn test_key_pool_rate_limit_persists_until_cleared() {
    let pool = KeyPool::new(
        vec![std::sync::Arc::new(std::sync::RwLock::new(make_key("k1")))],
        Box::new(FirstEligibleSelector),
    );

    pool.record_rate_limit("k1");
    {
        let key_arc = pool.get_key_mut("k1").unwrap();
        assert_eq!(
            key_arc.read().unwrap().health.status,
            KeyStatus::RateLimited
        );
    }

    // Recheck without clearing the counter: still rate-limited.
    pool.recheck_health();
    {
        let key_arc = pool.get_key_mut("k1").unwrap();
        assert_eq!(
            key_arc.read().unwrap().health.status,
            KeyStatus::RateLimited,
            "rate-limited status persists while the rate_limit_hits counter is > 0"
        );
    }

    // Clear the counter: recheck recovers.
    {
        let key_arc = pool.get_key_mut("k1").unwrap();
        key_arc.write().unwrap().health.rate_limit_hits_last_5m = 0;
    }
    pool.recheck_health();
    {
        let key_arc = pool.get_key_mut("k1").unwrap();
        assert_eq!(
            key_arc.read().unwrap().health.status,
            KeyStatus::Healthy,
            "key should recover to Healthy after rate limit counter is cleared"
        );
    }
}

/// When a pool has one `Disabled` key, `KeyPool::eligible_count()` returns 0
/// and `pool.select()` returns `None`.
#[test]
fn test_key_pool_all_disabled_returns_none() {
    let disabled = make_key_with_status("only-key", KeyStatus::Disabled);
    let pool = make_pool_weighted(vec![disabled]);
    let ctx = RequestContext::default();

    assert_eq!(
        pool.eligible_count(),
        0,
        "no eligible keys when all disabled"
    );
    assert!(
        pool.select(&ctx).is_none(),
        "select must return None when all keys are disabled"
    );
}

/// `RoundRobinSelector` with 3 healthy keys selected 300 times never returns
/// the same key more than 110 times (verifies it's not stuck on one key).
#[test]
fn test_round_robin_does_not_starve_any_key() {
    let pool_keys = vec![make_key("r1"), make_key("r2"), make_key("r3")];
    let pool = make_pool_round_robin(pool_keys);
    let ctx = RequestContext::default();

    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..300 {
        let id = pool.select(&ctx).unwrap().read().unwrap().id.clone();
        *counts.entry(id).or_insert(0) += 1;
    }

    for (id, count) in &counts {
        assert!(
            *count <= 110,
            "key {id} selected {count} times — round-robin must not starve keys"
        );
        assert!(
            *count >= 90,
            "key {id} selected {count} times — round-robin must visit all keys"
        );
    }
}
