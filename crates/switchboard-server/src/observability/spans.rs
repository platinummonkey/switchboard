//! Span attribute types and the [`ProxySpan`] builder.
//!
//! Every proxied LLM request creates a [`ProxySpan`] that accumulates
//! attributes during the request lifecycle.  Calling [`ProxySpan::finish`]
//! records all attributes onto the currently-active `tracing` span, which is
//! then exported to the OTel backend via `tracing-opentelemetry`.

use super::dd_llm_obs::attr;

// ── Span attributes ───────────────────────────────────────────────────────────

/// All OTel / Switchboard attributes for a single proxied LLM request.
///
/// Build via [`ProxySpan`]; record onto the active span with
/// [`SpanAttributes::record_on_current_span`].
#[derive(Debug, Default)]
pub struct SpanAttributes {
    // GenAI semantic conventions ─────────────────────────────────────────────
    /// Operation name: "chat" | "completion" | "embedding".
    pub operation_name: String,
    /// Provider name: "anthropic" | "openai" | "bedrock" | …
    pub provider_name: String,
    /// Model name sent in the request.
    pub request_model: String,
    /// Model name returned in the response (may differ from request).
    pub response_model: Option<String>,
    /// Input (prompt) token count.
    pub input_tokens: Option<u32>,
    /// Output (completion) token count.
    pub output_tokens: Option<u32>,
    /// Sampling temperature from the request.
    pub temperature: Option<f64>,
    /// Finish reasons (e.g. `["stop"]`).
    pub finish_reasons: Vec<String>,

    // Switchboard custom attributes ──────────────────────────────────────────
    /// Resolved user identifier.
    pub user_id: Option<String>,
    /// Resolved user team.
    pub user_team: Option<String>,
    /// Originating tool name (e.g. "cursor", "claude-code").
    pub tool_name: Option<String>,
    /// Key ID selected from the pool.
    pub key_id: Option<String>,
    /// Reason the model was selected.
    pub selection_reason: Option<String>,
    /// Pre-request guardrail action (e.g. "allow", "block", "redact").
    pub guardrail_pre_action: Option<String>,
    /// Post-response guardrail action.
    pub guardrail_post_action: Option<String>,
    /// Time-to-first-token in milliseconds (streaming requests only).
    pub time_to_first_token_ms: Option<f64>,
    /// Estimated cost in USD.
    pub cost_usd: Option<f64>,
}

impl SpanAttributes {
    /// Record all populated attributes onto the **currently active** `tracing`
    /// span.
    ///
    /// The `tracing-opentelemetry` layer will pick these up and forward them
    /// to the configured OTel exporter.  Fields that are `None` or empty are
    /// silently skipped.
    pub fn record_on_current_span(&self) {
        let span = tracing::Span::current();

        // GenAI semantic conventions
        span.record(attr::GEN_AI_OPERATION_NAME, self.operation_name.as_str());
        span.record(attr::GEN_AI_PROVIDER_NAME, self.provider_name.as_str());
        span.record(attr::GEN_AI_REQUEST_MODEL, self.request_model.as_str());

        if let Some(ref v) = self.response_model {
            span.record(attr::GEN_AI_RESPONSE_MODEL, v.as_str());
        }
        if let Some(v) = self.input_tokens {
            span.record(attr::GEN_AI_USAGE_INPUT_TOKENS, v);
        }
        if let Some(v) = self.output_tokens {
            span.record(attr::GEN_AI_USAGE_OUTPUT_TOKENS, v);
        }

        // Switchboard custom attributes
        if let Some(ref v) = self.user_id {
            span.record(attr::SWITCHBOARD_USER_ID, v.as_str());
        }
        if let Some(ref v) = self.user_team {
            span.record(attr::SWITCHBOARD_USER_TEAM, v.as_str());
        }
        if let Some(ref v) = self.tool_name {
            span.record(attr::SWITCHBOARD_TOOL, v.as_str());
        }
        if let Some(ref v) = self.key_id {
            span.record(attr::SWITCHBOARD_KEY_ID, v.as_str());
        }
        if let Some(ref v) = self.selection_reason {
            span.record(attr::SWITCHBOARD_SELECTION_REASON, v.as_str());
        }
        if let Some(ref v) = self.guardrail_pre_action {
            span.record(attr::SWITCHBOARD_GUARDRAIL_PRE, v.as_str());
        }
        if let Some(ref v) = self.guardrail_post_action {
            span.record(attr::SWITCHBOARD_GUARDRAIL_POST, v.as_str());
        }
        if let Some(v) = self.time_to_first_token_ms {
            span.record(attr::SWITCHBOARD_TTFT_MS, v);
        }
        if let Some(v) = self.cost_usd {
            span.record(attr::SWITCHBOARD_COST_USD, v);
        }
    }
}

// ── ProxySpan builder ─────────────────────────────────────────────────────────

/// A builder that accumulates span attributes throughout the lifecycle of a
/// single proxied LLM request.
///
/// # Usage
/// ```rust,ignore
/// let mut span = ProxySpan::start("chat", "anthropic", "claude-sonnet-4-20250514");
/// span.set_user("alice@example.com", Some("platform"));
/// span.set_tokens(512, 256);
/// span.finish(); // records all attributes on the current tracing span
/// ```
pub struct ProxySpan {
    attrs: SpanAttributes,
    start: std::time::Instant,
}

impl ProxySpan {
    /// Create a new span builder and record the start time.
    ///
    /// # Arguments
    /// * `operation` – GenAI operation name ("chat", "completion", …).
    /// * `provider`  – Provider name ("anthropic", "openai", …).
    /// * `model`     – Model identifier from the request.
    pub fn start(operation: &str, provider: &str, model: &str) -> Self {
        Self {
            attrs: SpanAttributes {
                operation_name: operation.to_string(),
                provider_name: provider.to_string(),
                request_model: model.to_string(),
                ..SpanAttributes::default()
            },
            start: std::time::Instant::now(),
        }
    }

    /// Set the resolved user identity.
    pub fn set_user(&mut self, id: &str, team: Option<&str>) {
        self.attrs.user_id = Some(id.to_string());
        self.attrs.user_team = team.map(|t| t.to_string());
    }

    /// Set the key that was selected from the pool.
    pub fn set_key(&mut self, key_id: &str) {
        self.attrs.key_id = Some(key_id.to_string());
    }

    /// Set the reason the model was selected.
    pub fn set_selection_reason(&mut self, reason: &str) {
        self.attrs.selection_reason = Some(reason.to_string());
    }

    /// Set the pre-request guardrail action.
    pub fn set_guardrail_pre(&mut self, action: &str) {
        self.attrs.guardrail_pre_action = Some(action.to_string());
    }

    /// Set the post-response guardrail action.
    pub fn set_guardrail_post(&mut self, action: &str) {
        self.attrs.guardrail_post_action = Some(action.to_string());
    }

    /// Set the token usage counts.
    pub fn set_tokens(&mut self, input: u32, output: u32) {
        self.attrs.input_tokens = Some(input);
        self.attrs.output_tokens = Some(output);
    }

    /// Set the time-to-first-token in milliseconds.
    pub fn set_ttft(&mut self, ms: f64) {
        self.attrs.time_to_first_token_ms = Some(ms);
    }

    /// Finalise the span: compute elapsed time, derive cost if tokens are
    /// available, then record all attributes on the currently-active
    /// `tracing` span.
    pub fn finish(mut self) {
        // If we have token data and no cost has been set, derive cost.
        if self.attrs.cost_usd.is_none() {
            if let (Some(input), Some(output)) = (self.attrs.input_tokens, self.attrs.output_tokens)
            {
                self.attrs.cost_usd = Some(super::dd_llm_obs::calculate_cost(
                    &self.attrs.request_model,
                    input,
                    output,
                ));
            }
        }

        // Record elapsed as a latency metric event.
        let elapsed_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!(
            target: "switchboard.span",
            elapsed_ms = elapsed_ms,
            operation = %self.attrs.operation_name,
            provider = %self.attrs.provider_name,
            model = %self.attrs.request_model,
            "proxy span finished"
        );

        self.attrs.record_on_current_span();
    }
}

// ── Convenience macro ─────────────────────────────────────────────────────────

/// Create an `info`-level `tracing` span for a switchboard proxy operation.
///
/// # Example
/// ```rust,ignore
/// let _span = proxy_span!("chat");
/// ```
#[macro_export]
macro_rules! proxy_span {
    ($name:expr) => {
        tracing::info_span!("switchboard.proxy", otel.name = $name)
    };
    ($name:expr, $($field:tt)*) => {
        tracing::info_span!("switchboard.proxy", otel.name = $name, $($field)*)
    };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_attributes_default() {
        let attrs = SpanAttributes::default();
        assert!(attrs.operation_name.is_empty());
        assert!(attrs.provider_name.is_empty());
        assert!(attrs.request_model.is_empty());
        assert!(attrs.response_model.is_none());
        assert!(attrs.input_tokens.is_none());
        assert!(attrs.output_tokens.is_none());
        assert!(attrs.temperature.is_none());
        assert!(attrs.finish_reasons.is_empty());
        assert!(attrs.user_id.is_none());
        assert!(attrs.user_team.is_none());
        assert!(attrs.tool_name.is_none());
        assert!(attrs.key_id.is_none());
        assert!(attrs.selection_reason.is_none());
        assert!(attrs.guardrail_pre_action.is_none());
        assert!(attrs.guardrail_post_action.is_none());
        assert!(attrs.time_to_first_token_ms.is_none());
        assert!(attrs.cost_usd.is_none());
    }

    #[test]
    fn test_proxy_span_start_and_finish() {
        // Verify that creating and finishing a ProxySpan does not panic even
        // when no tracing subscriber is active.
        let mut span = ProxySpan::start("chat", "anthropic", "claude-sonnet-4-20250514");
        span.set_user("alice@example.com", Some("platform"));
        span.set_key("key-001");
        span.set_selection_reason("config");
        span.set_guardrail_pre("allow");
        span.set_guardrail_post("allow");
        span.set_tokens(512, 256);
        span.set_ttft(42.5);
        span.finish(); // must not panic
    }

    #[test]
    fn test_proxy_span_finish_without_tokens_no_cost() {
        // If no tokens are provided, cost_usd should not be set (no panic).
        let span = ProxySpan::start("chat", "openai", "gpt-4o");
        span.finish(); // must not panic
    }

    #[test]
    fn test_proxy_span_finish_auto_cost() {
        // finish() should compute cost automatically when tokens are known.
        let mut span = ProxySpan::start("chat", "openai", "gpt-4o");
        span.set_tokens(1000, 500);
        // finish() will derive cost and record on (noop) current span — no panic.
        span.finish();
    }

    #[test]
    fn test_span_attributes_record_on_current_span_no_panic() {
        // With no active subscriber the record calls are silently ignored.
        let mut attrs = SpanAttributes {
            operation_name: "chat".into(),
            provider_name: "openai".into(),
            request_model: "gpt-4o".into(),
            ..SpanAttributes::default()
        };
        attrs.response_model = Some("gpt-4o".into());
        attrs.input_tokens = Some(100);
        attrs.output_tokens = Some(50);
        attrs.user_id = Some("bob".into());
        attrs.user_team = Some("eng".into());
        attrs.tool_name = Some("cursor".into());
        attrs.key_id = Some("k1".into());
        attrs.selection_reason = Some("config".into());
        attrs.guardrail_pre_action = Some("allow".into());
        attrs.guardrail_post_action = Some("allow".into());
        attrs.time_to_first_token_ms = Some(10.0);
        attrs.cost_usd = Some(0.001);
        attrs.record_on_current_span(); // must not panic
    }
}
