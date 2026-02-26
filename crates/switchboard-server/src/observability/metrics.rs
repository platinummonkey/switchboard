//! Lightweight metrics emission via structured `tracing` events.
//!
//! Each method emits a `tracing::info!` event targeted at
//! `"switchboard.metrics"` with structured fields that can be consumed by a
//! metrics scraper or forwarded via the OpenTelemetry SDK.
//!
//! # Design note
//! We deliberately avoid adding Prometheus or another metrics crate as a
//! dependency in this phase.  The structured fields produced here are
//! sufficient for DD Logs-based metric extraction and can be upgraded to
//! native OTel metrics in a future phase.

/// Metric name constants for consistent field naming.
pub mod names {
    pub const REQUESTS_TOTAL: &str = "switchboard.requests.total";
    pub const TOKENS_INPUT_TOTAL: &str = "switchboard.tokens.input.total";
    pub const TOKENS_OUTPUT_TOTAL: &str = "switchboard.tokens.output.total";
    pub const GUARDRAIL_BLOCKED_TOTAL: &str = "switchboard.guardrail.blocked.total";
    pub const LATENCY_MS: &str = "switchboard.latency.ms";
    pub const COST_USD: &str = "switchboard.cost.usd";
}

/// Emits structured `tracing` events that serve as lightweight metrics.
pub struct Metrics;

impl Metrics {
    /// Record an inbound request with its final HTTP status code.
    pub fn record_request(provider: &str, model: &str, status: u16) {
        tracing::info!(
            target: "switchboard.metrics",
            metric = names::REQUESTS_TOTAL,
            provider = provider,
            model = model,
            status = status,
            value = 1u64,
            "request"
        );
    }

    /// Record token usage for a completed request.
    pub fn record_tokens(provider: &str, model: &str, input: u32, output: u32) {
        tracing::info!(
            target: "switchboard.metrics",
            metric = names::TOKENS_INPUT_TOTAL,
            provider = provider,
            model = model,
            value = input,
            "input_tokens"
        );
        tracing::info!(
            target: "switchboard.metrics",
            metric = names::TOKENS_OUTPUT_TOTAL,
            provider = provider,
            model = model,
            value = output,
            "output_tokens"
        );
    }

    /// Record a guardrail block event.
    pub fn record_guardrail_blocked(engine: &str, rule: &str, user: &str, team: &str) {
        tracing::info!(
            target: "switchboard.metrics",
            metric = names::GUARDRAIL_BLOCKED_TOTAL,
            engine = engine,
            rule = rule,
            user = user,
            team = team,
            value = 1u64,
            "guardrail_blocked"
        );
    }

    /// Record end-to-end proxy latency in milliseconds.
    pub fn record_latency_ms(provider: &str, model: &str, latency_ms: f64) {
        tracing::info!(
            target: "switchboard.metrics",
            metric = names::LATENCY_MS,
            provider = provider,
            model = model,
            value = latency_ms,
            "latency"
        );
    }

    /// Record the estimated cost in USD for a completed request.
    pub fn record_cost(provider: &str, model: &str, cost_usd: f64) {
        tracing::info!(
            target: "switchboard.metrics",
            metric = names::COST_USD,
            provider = provider,
            model = model,
            value = cost_usd,
            "cost"
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_record_request_no_panic() {
        // Ensure none of the metric methods panic.  We don't need a subscriber
        // set up because tracing is a no-op when there is no subscriber.
        Metrics::record_request("openai", "gpt-4o", 200);
        Metrics::record_request("anthropic", "claude-sonnet-4-20250514", 500);
        Metrics::record_tokens("openai", "gpt-4o", 100, 50);
        Metrics::record_guardrail_blocked("regex", "pii-rule", "alice", "platform");
        Metrics::record_latency_ms("openai", "gpt-4o", 123.4);
        Metrics::record_cost("openai", "gpt-4o", 0.001_23);
    }
}
