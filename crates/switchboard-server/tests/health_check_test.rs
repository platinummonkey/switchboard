//! Integration tests for ProviderHealthChecker and spawn_health_checks.
//!
//! NOTE: spawn_health_checks is not currently called by run_server.
//! These tests exercise the health check loop in isolation, directly calling
//! the machinery without going through the full proxy stack.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use http::{HeaderName, HeaderValue};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use switchboard_server::auth::UpstreamCredentials;
use switchboard_server::key_pool::health::KeyHealth;
use switchboard_server::key_pool::pool::KeyPool;
use switchboard_server::key_pool::provider::{KeySource, PooledKey};
use switchboard_server::key_pool::selector::WeightedRandomSelector;
use switchboard_server::providers::{OpenAiProvider, ProviderRegistry};
use switchboard_server::routing::ProviderHealthChecker;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_static_key(header_name: &'static str, header_value: &'static str) -> PooledKey {
    PooledKey {
        id: "health-check-key".into(),
        credentials: UpstreamCredentials {
            header_name: HeaderName::from_static(header_name),
            header_value: HeaderValue::from_static(header_value),
            expires_at: None,
        },
        weight: 1.0,
        source: KeySource::Static,
        health: KeyHealth::default(),
    }
}

fn make_pool(key: PooledKey) -> Arc<KeyPool> {
    Arc::new(KeyPool::new(vec![key], Box::new(WeightedRandomSelector)))
}

/// Build a ProviderRegistry and key_pools map with a single OpenAI provider
/// pointing at `base_url`.
fn build_openai_setup(
    base_url: &str,
) -> (Arc<ProviderRegistry>, Arc<HashMap<String, Arc<KeyPool>>>) {
    let provider = Arc::new(OpenAiProvider::new_named(
        "openai",
        base_url,
        vec!["gpt-4o".into()],
        Duration::from_secs(5),
    ));

    let mut registry = ProviderRegistry::new();
    registry.register("openai", provider);

    let mut key_pools: HashMap<String, Arc<KeyPool>> = HashMap::new();
    key_pools.insert(
        "openai".into(),
        make_pool(make_static_key("authorization", "Bearer sk-test")),
    );

    (Arc::new(registry), Arc::new(key_pools))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// After a successful GET /v1/models the checker marks the provider healthy.
#[tokio::test]
async fn test_health_checker_marks_provider_healthy_after_successful_check() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [{"id": "gpt-4o", "object": "model"}]
        })))
        // Allow many calls — the loop fires every 50 ms.
        .mount(&mock_server)
        .await;

    let (registry, key_pools) = build_openai_setup(&mock_server.uri());

    // Clone an Arc reference before spawning so we can query status afterwards.
    let checker = Arc::new(ProviderHealthChecker::new());
    let checker_for_query = Arc::clone(&checker);

    checker.spawn_health_checks(registry, key_pools, Duration::from_millis(50));

    // Give the loop enough time to run at least two check cycles.
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(
        checker_for_query.is_healthy("openai"),
        "provider should be marked healthy after a successful /v1/models response"
    );

    // Verify the mock was actually called.
    let received = mock_server.received_requests().await.unwrap();
    assert!(
        !received.is_empty(),
        "mock should have received at least one GET /v1/models request"
    );
}

/// After a 503 response the checker marks the provider unhealthy.
#[tokio::test]
async fn test_health_checker_marks_provider_unhealthy_after_failing_check() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503).set_body_string("Service Unavailable"))
        .mount(&mock_server)
        .await;

    let (registry, key_pools) = build_openai_setup(&mock_server.uri());

    let checker = Arc::new(ProviderHealthChecker::new());
    let checker_for_query = Arc::clone(&checker);

    checker.spawn_health_checks(registry, key_pools, Duration::from_millis(50));

    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(
        !checker_for_query.is_healthy("openai"),
        "provider should be marked unhealthy after a 503 response"
    );

    let received = mock_server.received_requests().await.unwrap();
    assert!(
        !received.is_empty(),
        "mock should have received at least one GET /v1/models request"
    );
}

/// The checker recovers: starts with 503 responses, switches to 200, and the
/// health status eventually flips back to healthy.
#[tokio::test]
async fn test_health_checker_recovers_after_failures() {
    let mock_server = MockServer::start().await;

    // First two requests return 503.
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(2)
        .mount(&mock_server)
        .await;

    // All subsequent requests return 200.
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [{"id": "gpt-4o", "object": "model"}]
        })))
        .mount(&mock_server)
        .await;

    let (registry, key_pools) = build_openai_setup(&mock_server.uri());

    let checker = Arc::new(ProviderHealthChecker::new());
    let checker_for_query = Arc::clone(&checker);

    // Use a short interval so the test finishes quickly.
    checker.spawn_health_checks(registry, key_pools, Duration::from_millis(50));

    // Wait long enough for both the failure cycles and the recovery cycle.
    tokio::time::sleep(Duration::from_millis(400)).await;

    assert!(
        checker_for_query.is_healthy("openai"),
        "provider should recover to healthy after 503 responses are replaced by 200"
    );
}

/// A provider that has never been checked defaults to healthy (optimistic).
#[tokio::test]
async fn test_health_checker_unknown_provider_is_healthy() {
    // No providers registered, no background task spawned.
    let checker = ProviderHealthChecker::new();

    assert!(
        checker.is_healthy("nonexistent"),
        "unknown providers should be treated as healthy by default"
    );
}

/// The checker correctly handles a registry with no key pool for a provider:
/// the provider is skipped and no panic occurs.
#[tokio::test]
async fn test_health_checker_skips_provider_with_no_key_pool() {
    let mock_server = MockServer::start().await;

    // Mount nothing — if any request arrives the test will see unexpected calls.
    let (registry, _key_pools_with_openai) = build_openai_setup(&mock_server.uri());

    // Pass an empty key pool map so the provider has no pool available.
    let empty_key_pools: Arc<HashMap<String, Arc<KeyPool>>> = Arc::new(HashMap::new());

    let checker = Arc::new(ProviderHealthChecker::new());
    let checker_for_query = Arc::clone(&checker);

    checker.spawn_health_checks(registry, empty_key_pools, Duration::from_millis(50));

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Provider was never checked, so it defaults to healthy.
    assert!(
        checker_for_query.is_healthy("openai"),
        "unchecked provider should default to healthy"
    );

    // Crucially, no request was sent to the mock.
    let received = mock_server.received_requests().await.unwrap();
    assert!(
        received.is_empty(),
        "no health check request should be sent when the key pool is missing"
    );
}

/// The record/is_healthy API works correctly in isolation (no background task).
#[tokio::test]
async fn test_health_checker_record_and_query_without_background_task() {
    let checker = ProviderHealthChecker::new();

    // Initially healthy.
    assert!(checker.is_healthy("anthropic"));

    // Mark as unhealthy.
    checker.record("anthropic", false);
    assert!(!checker.is_healthy("anthropic"));

    // Mark as healthy again.
    checker.record("anthropic", true);
    assert!(checker.is_healthy("anthropic"));
}
