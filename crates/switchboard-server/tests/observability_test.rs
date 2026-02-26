//! Integration tests for switchboard-server observability.
//!
//! Covers:
//! - `init_tracing` disabled (no-op) path
//! - `calculate_cost` pricing for known and unknown models
//! - `input_cost_per_million` / `output_cost_per_million` for all 6 known models
//! - `ProxySpan` lifecycle (start → mutate → finish)
//! - `SpanAttributes::record_on_current_span` with no subscriber
//! - `Metrics` all record methods (no panic)
//! - `attr` constant correctness (non-empty, correct prefix)
//! - `ml_app_name` returns non-empty string

use switchboard_server::config::observability::ObservabilityConfig;
use switchboard_server::observability::dd_llm_obs::{
    attr, calculate_cost, input_cost_per_million, ml_app_name, output_cost_per_million,
};
use switchboard_server::observability::init_tracing;
use switchboard_server::observability::metrics::Metrics;
use switchboard_server::observability::spans::{ProxySpan, SpanAttributes};

// ── init_tracing ──────────────────────────────────────────────────────────────

/// When OTel is disabled the function must return Ok and the guard must hold
/// no provider.  Dropping the guard must not panic.
#[test]
fn test_init_tracing_disabled_is_noop() {
    let config = ObservabilityConfig {
        enabled: false,
        ..ObservabilityConfig::default()
    };
    let result = init_tracing(&config);
    assert!(result.is_ok(), "disabled init_tracing should return Ok");
    let guard = result.unwrap();
    // Drop must not panic — the guard with None provider is a true no-op.
    drop(guard);
}

// ── calculate_cost ────────────────────────────────────────────────────────────

/// 1 M input + 1 M output for claude-sonnet-4-20250514:
///   input:  1_000_000 / 1_000_000 * 3.00 = 3.00
///   output: 1_000_000 / 1_000_000 * 15.00 = 15.00
///   total: 18.00
#[test]
fn test_cost_claude_sonnet_calculation() {
    let cost = calculate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000);
    let expected = 3.00_f64 + 15.00_f64;
    assert!(
        (cost - expected).abs() < 1e-9,
        "expected {expected} but got {cost}"
    );
}

/// claude-opus-4-20250514 must cost more than claude-sonnet-4-20250514 for the
/// same token counts.
#[test]
fn test_cost_claude_opus_higher_than_sonnet() {
    let sonnet = calculate_cost("claude-sonnet-4-20250514", 1000, 1000);
    let opus = calculate_cost("claude-opus-4-20250514", 1000, 1000);
    assert!(
        opus > sonnet,
        "opus cost {opus} should exceed sonnet cost {sonnet}"
    );
}

/// An unknown model must fall back to the default pricing and return a value
/// greater than zero.
#[test]
fn test_cost_unknown_model_uses_fallback() {
    let cost = calculate_cost("unknown-model-xyz", 1000, 1000);
    assert!(
        cost > 0.0,
        "unknown model fallback cost should be positive, got {cost}"
    );
}

/// Zero tokens must produce zero cost regardless of model.
#[test]
fn test_cost_zero_tokens() {
    let cost = calculate_cost("gpt-4o", 0, 0);
    assert_eq!(cost, 0.0, "zero tokens must yield zero cost");
}

/// Every known model must have positive input AND output cost per million.
#[test]
fn test_all_known_models_have_nonzero_cost() {
    let models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-4-5-20251001",
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
    ];
    for model in models {
        let input = input_cost_per_million(model);
        let output = output_cost_per_million(model);
        assert!(
            input > 0.0,
            "input_cost_per_million({model}) = {input} should be > 0"
        );
        assert!(
            output > 0.0,
            "output_cost_per_million({model}) = {output} should be > 0"
        );
    }
}

/// gpt-4o pricing: $5.00 / M input, $15.00 / M output.
#[test]
fn test_cost_gpt4o_pricing() {
    let input = input_cost_per_million("gpt-4o");
    let output = output_cost_per_million("gpt-4o");
    assert!(
        (input - 5.0).abs() < f64::EPSILON,
        "gpt-4o input cost per million should be 5.0, got {input}"
    );
    assert!(
        (output - 15.0).abs() < f64::EPSILON,
        "gpt-4o output cost per million should be 15.0, got {output}"
    );
}

// ── ProxySpan ─────────────────────────────────────────────────────────────────

/// A full ProxySpan lifecycle must not panic even when no tracing subscriber is
/// active.  Cost is auto-derived from tokens in `finish()`.
#[test]
fn test_proxy_span_lifecycle() {
    let mut span = ProxySpan::start("chat", "anthropic", "claude-sonnet-4-20250514");
    span.set_user("alice", Some("platform"));
    span.set_tokens(1500, 800);
    span.set_ttft(340.0);
    // finish() derives cost automatically — no panic expected.
    span.finish();
}

// ── SpanAttributes ────────────────────────────────────────────────────────────

/// `SpanAttributes::record_on_current_span` must not panic when no tracing
/// subscriber is installed.  The `tracing` crate silently discards the records.
#[test]
fn test_span_attributes_record_noop_without_subscriber() {
    SpanAttributes::default().record_on_current_span();
}

// ── Metrics ───────────────────────────────────────────────────────────────────

/// Every `Metrics::record_*` method must not panic.  No subscriber is needed
/// because `tracing` discards events when none is installed.
#[test]
fn test_metrics_record_all_methods_no_panic() {
    Metrics::record_request("anthropic", "claude-sonnet-4-20250514", 200);
    Metrics::record_request("openai", "gpt-4o", 400);
    Metrics::record_tokens("anthropic", "claude-sonnet-4-20250514", 512, 256);
    Metrics::record_tokens("openai", "gpt-4o", 1000, 500);
    Metrics::record_guardrail_blocked("regex", "pii-ssn", "alice@example.com", "platform");
    Metrics::record_latency_ms("anthropic", "claude-sonnet-4-20250514", 123.4);
    Metrics::record_latency_ms("openai", "gpt-4o", 0.9);
    Metrics::record_cost("anthropic", "claude-sonnet-4-20250514", 0.002_25);
    Metrics::record_cost("openai", "gpt-4o", 0.000_05);
}

// ── attr constants ────────────────────────────────────────────────────────────

/// All 15 attribute name constants must be non-empty and carry the expected
/// `gen_ai.` or `switchboard.` prefix.
#[test]
fn test_attr_constants_nonempty() {
    let gen_ai_attrs = [
        attr::GEN_AI_OPERATION_NAME,
        attr::GEN_AI_PROVIDER_NAME,
        attr::GEN_AI_REQUEST_MODEL,
        attr::GEN_AI_RESPONSE_MODEL,
        attr::GEN_AI_USAGE_INPUT_TOKENS,
        attr::GEN_AI_USAGE_OUTPUT_TOKENS,
    ];
    let switchboard_attrs = [
        attr::SWITCHBOARD_USER_ID,
        attr::SWITCHBOARD_USER_TEAM,
        attr::SWITCHBOARD_TOOL,
        attr::SWITCHBOARD_KEY_ID,
        attr::SWITCHBOARD_SELECTION_REASON,
        attr::SWITCHBOARD_GUARDRAIL_PRE,
        attr::SWITCHBOARD_GUARDRAIL_POST,
        attr::SWITCHBOARD_TTFT_MS,
        attr::SWITCHBOARD_COST_USD,
    ];

    for name in gen_ai_attrs {
        assert!(!name.is_empty(), "attr constant must be non-empty");
        assert!(
            name.starts_with("gen_ai."),
            "gen_ai attr '{name}' must start with 'gen_ai.'"
        );
    }
    for name in switchboard_attrs {
        assert!(!name.is_empty(), "attr constant must be non-empty");
        assert!(
            name.starts_with("switchboard."),
            "switchboard attr '{name}' must start with 'switchboard.'"
        );
    }
}

// ── ml_app_name ───────────────────────────────────────────────────────────────

/// `ml_app_name` must return a non-empty string derived from the service name.
#[test]
fn test_ml_app_name_returns_service_name() {
    let name = ml_app_name("switchboard");
    assert!(
        !name.is_empty(),
        "ml_app_name must return a non-empty string"
    );
}

// ── OtelGuard ─────────────────────────────────────────────────────────────────

/// A guard returned from `init_tracing(disabled)` must drop without panicking.
#[test]
fn test_otel_guard_drop_no_provider_no_panic() {
    let config = ObservabilityConfig {
        enabled: false,
        ..ObservabilityConfig::default()
    };
    let guard = init_tracing(&config).expect("disabled init_tracing must succeed");
    drop(guard);
}
