//! Proxy overhead benchmarks.
//!
//! Measures the cost of hot-path transformations and key-pool selection that
//! run on every request, independent of network I/O.
//!
//! Run with:  cargo bench -p switchboard-server

use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};

use switchboard_common::types::{Message, MessageContent, ProxiedResponse, Role, Usage};
use switchboard_server::proxy::transform::{
    anthropic_to_proxied, openai_to_proxied, proxied_to_anthropic, proxied_to_openai,
};

// ── Sample payloads ────────────────────────────────────────────────────────────

fn openai_chat_body() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Write a Rust function that adds two numbers."}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": false
    })
}

fn anthropic_chat_body() -> serde_json::Value {
    serde_json::json!({
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Write a Rust function that adds two numbers."}],
        "system": "You are a helpful assistant.",
        "max_tokens": 1024,
        "stream": false
    })
}

fn sample_proxied_response() -> ProxiedResponse {
    ProxiedResponse {
        model: "gpt-4o".into(),
        content: "fn add(a: i32, b: i32) -> i32 { a + b }".into(),
        tool_calls: None,
        finish_reason: Some("stop".into()),
        usage: Some(Usage {
            input_tokens: 25,
            output_tokens: 18,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }),
    }
}

// ── Transform benchmarks ───────────────────────────────────────────────────────

fn bench_openai_to_proxied(c: &mut Criterion) {
    let body = openai_chat_body();
    c.bench_function("openai_to_proxied", |b| {
        b.iter(|| openai_to_proxied(&body).unwrap())
    });
}

fn bench_proxied_to_openai(c: &mut Criterion) {
    let resp = sample_proxied_response();
    c.bench_function("proxied_to_openai", |b| b.iter(|| proxied_to_openai(&resp)));
}

fn bench_anthropic_to_proxied(c: &mut Criterion) {
    let body = anthropic_chat_body();
    c.bench_function("anthropic_to_proxied", |b| {
        b.iter(|| anthropic_to_proxied(&body).unwrap())
    });
}

fn bench_proxied_to_anthropic(c: &mut Criterion) {
    let resp = sample_proxied_response();
    c.bench_function("proxied_to_anthropic", |b| {
        b.iter(|| proxied_to_anthropic(&resp))
    });
}

// ── Model selection benchmark ──────────────────────────────────────────────────

fn bench_model_selector_static(c: &mut Criterion) {
    use switchboard_common::types::RequestContext;
    use switchboard_server::config::model_selection::ModelSelectionConfig;
    use switchboard_server::routing::selector::ModelSelector;

    let cfg = ModelSelectionConfig {
        mode: "static".into(),
        model: Some("claude-sonnet-4-20250514".into()),
        ..ModelSelectionConfig::default()
    };
    let selector = ModelSelector::new(cfg);
    let ctx = RequestContext::new();

    c.bench_function("model_selector_static", |b| {
        b.iter(|| selector.select(&ctx))
    });
}

fn bench_model_selector_dynamic(c: &mut Criterion) {
    use switchboard_common::types::RequestContext;
    use switchboard_server::config::model_selection::ModelSelectionConfig;
    use switchboard_server::routing::selector::ModelSelector;

    let cfg = ModelSelectionConfig {
        mode: "dynamic".into(),
        fallback: Some("claude-sonnet-4-20250514".into()),
        ..ModelSelectionConfig::default()
    };
    let selector = ModelSelector::new(cfg);
    let ctx = RequestContext::new();

    c.bench_function("model_selector_dynamic", |b| {
        b.iter(|| selector.select(&ctx))
    });
}

// ── Semantic classifier benchmark ──────────────────────────────────────────────

fn bench_heuristic_classifier(c: &mut Criterion) {
    use switchboard_server::routing::semantic::estimate_tokens;
    use switchboard_server::routing::{
        ClassificationInput, HeuristicClassifier, SemanticClassifier,
    };

    let messages = vec![
        Message {
            role: Role::System,
            content: MessageContent::Text("You are a Rust programming assistant.".into()),
            tool_call_id: None,
            tool_calls: None,
        },
        Message {
            role: Role::User,
            content: MessageContent::Text(
                "Write a function that parses JSON and returns the first key.".into(),
            ),
            tool_call_id: None,
            tool_calls: None,
        },
    ];
    let input = ClassificationInput {
        estimated_tokens: estimate_tokens(&messages),
        messages,
        tools: None,
    };
    let classifier = HeuristicClassifier::new();

    // The classifier is async but internally sync — use block_in_place.
    c.bench_function("heuristic_classifier", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| classifier.classify(&input))
    });
}

// ── Key pool selection benchmark ───────────────────────────────────────────────

fn bench_key_pool_select(c: &mut Criterion) {
    use http::{HeaderName, HeaderValue};
    use switchboard_common::types::RequestContext;
    use switchboard_server::auth::UpstreamCredentials;
    use switchboard_server::key_pool::health::KeyHealth;
    use switchboard_server::key_pool::pool::KeyPool;
    use switchboard_server::key_pool::provider::{KeySource, PooledKey};
    use switchboard_server::key_pool::selector::WeightedRandomSelector;

    let keys: Vec<Arc<std::sync::RwLock<PooledKey>>> = (0..10)
        .map(|i| {
            Arc::new(std::sync::RwLock::new(PooledKey {
                id: format!("key-{i}"),
                credentials: UpstreamCredentials {
                    header_name: HeaderName::from_static("authorization"),
                    header_value: HeaderValue::from_static("Bearer sk-bench"),
                    expires_at: None,
                },
                weight: 1.0,
                source: KeySource::Static,
                health: KeyHealth::default(),
            }))
        })
        .collect();

    let pool = KeyPool::new(keys, Box::new(WeightedRandomSelector));
    let ctx = RequestContext::new();

    c.bench_function("key_pool_select_10_keys", |b| b.iter(|| pool.select(&ctx)));
}

criterion_group!(
    benches,
    bench_openai_to_proxied,
    bench_proxied_to_openai,
    bench_anthropic_to_proxied,
    bench_proxied_to_anthropic,
    bench_model_selector_static,
    bench_model_selector_dynamic,
    bench_heuristic_classifier,
    bench_key_pool_select,
);
criterion_main!(benches);
