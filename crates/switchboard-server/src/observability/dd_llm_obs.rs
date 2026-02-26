//! Datadog LLM Observability helpers: cost calculation and GenAI semantic
//! convention attribute name constants.

// ── Cost tables ───────────────────────────────────────────────────────────────

/// Cost per 1 million *input* tokens in USD for a known model.
/// Returns a fallback value for unknown models.
pub fn input_cost_per_million(model: &str) -> f64 {
    match model {
        "claude-sonnet-4-20250514" => 3.00,
        "claude-opus-4-20250514" => 15.00,
        "claude-haiku-4-5-20251001" => 0.80,
        "gpt-4o" => 5.00,
        "gpt-4o-mini" => 0.15,
        "o1" => 15.00,
        _ => 5.00, // fallback
    }
}

/// Cost per 1 million *output* tokens in USD for a known model.
/// Returns a fallback value for unknown models.
pub fn output_cost_per_million(model: &str) -> f64 {
    match model {
        "claude-sonnet-4-20250514" => 15.00,
        "claude-opus-4-20250514" => 75.00,
        "claude-haiku-4-5-20251001" => 4.00,
        "gpt-4o" => 15.00,
        "gpt-4o-mini" => 0.60,
        "o1" => 60.00,
        _ => 15.00, // fallback
    }
}

/// Calculate total cost in USD for a completed request.
///
/// # Arguments
/// * `model`         – Model identifier string.
/// * `input_tokens`  – Number of input (prompt) tokens consumed.
/// * `output_tokens` – Number of output (completion) tokens produced.
pub fn calculate_cost(model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
    let input_cost = input_cost_per_million(model) * (input_tokens as f64) / 1_000_000.0;
    let output_cost = output_cost_per_million(model) * (output_tokens as f64) / 1_000_000.0;
    input_cost + output_cost
}

// ── DD-specific helpers ───────────────────────────────────────────────────────

/// Build the `ml_app` attribute value used by Datadog LLM Observability.
///
/// The value is derived from the service name so that traces are grouped
/// correctly in the DD LLM Obs product.
pub fn ml_app_name(service_name: &str) -> String {
    service_name.to_string()
}

// ── Attribute name constants ──────────────────────────────────────────────────

/// GenAI semantic convention attribute key names and Switchboard custom keys.
pub mod attr {
    /// `gen_ai.operation.name` – e.g. "chat", "completion", "embedding".
    pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";

    /// `gen_ai.system` – provider name, e.g. "anthropic", "openai".
    /// Note: the OpenTelemetry GenAI spec uses `gen_ai.system` for the provider.
    pub const GEN_AI_PROVIDER_NAME: &str = "gen_ai.system";

    /// `gen_ai.request.model` – model name sent in the request.
    pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";

    /// `gen_ai.response.model` – model name returned in the response.
    pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";

    /// `gen_ai.usage.input_tokens` – input token count.
    pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";

    /// `gen_ai.usage.output_tokens` – output token count.
    pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";

    /// `switchboard.user.id` – resolved user identifier.
    pub const SWITCHBOARD_USER_ID: &str = "switchboard.user.id";

    /// `switchboard.user.team` – resolved user team.
    pub const SWITCHBOARD_USER_TEAM: &str = "switchboard.user.team";

    /// `switchboard.tool` – tool name (e.g. "cursor", "claude-code").
    pub const SWITCHBOARD_TOOL: &str = "switchboard.tool";

    /// `switchboard.key_pool.key_id` – key selected from the pool.
    pub const SWITCHBOARD_KEY_ID: &str = "switchboard.key_pool.key_id";

    /// `switchboard.model.selection_reason` – why this model was chosen.
    pub const SWITCHBOARD_SELECTION_REASON: &str = "switchboard.model.selection_reason";

    /// `switchboard.guardrail.pre_action` – action taken by pre-request guardrail.
    pub const SWITCHBOARD_GUARDRAIL_PRE: &str = "switchboard.guardrail.pre_action";

    /// `switchboard.guardrail.post_action` – action taken by post-response guardrail.
    pub const SWITCHBOARD_GUARDRAIL_POST: &str = "switchboard.guardrail.post_action";

    /// `switchboard.time_to_first_token_ms` – TTFT in milliseconds for streaming.
    pub const SWITCHBOARD_TTFT_MS: &str = "switchboard.time_to_first_token_ms";

    /// `switchboard.cost_usd` – estimated request cost in USD.
    pub const SWITCHBOARD_COST_USD: &str = "switchboard.cost_usd";
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_claude_sonnet() {
        // 1000 input + 500 output tokens
        // input: 1000 / 1_000_000 * 3.00 = 0.000_003 USD
        // output: 500 / 1_000_000 * 15.00 = 0.000_007_5 USD
        let cost = calculate_cost("claude-sonnet-4-20250514", 1000, 500);
        let expected = 1000.0 / 1_000_000.0 * 3.00 + 500.0 / 1_000_000.0 * 15.00;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    #[test]
    fn test_cost_claude_opus() {
        // opus is more expensive than sonnet
        let sonnet_cost = calculate_cost("claude-sonnet-4-20250514", 1000, 500);
        let opus_cost = calculate_cost("claude-opus-4-20250514", 1000, 500);
        assert!(
            opus_cost > sonnet_cost,
            "opus ({opus_cost}) should cost more than sonnet ({sonnet_cost})"
        );
    }

    #[test]
    fn test_cost_gpt4o() {
        let cost = calculate_cost("gpt-4o", 1000, 500);
        let expected = 1000.0 / 1_000_000.0 * 5.00 + 500.0 / 1_000_000.0 * 15.00;
        assert!(
            (cost - expected).abs() < 1e-12,
            "got {cost}, expected {expected}"
        );
    }

    #[test]
    fn test_cost_unknown_model_uses_fallback() {
        // fallback: 5.00 / M input, 15.00 / M output — same as gpt-4o
        let unknown_cost = calculate_cost("totally-unknown-model-xyz", 1000, 500);
        let fallback_expected = 1000.0 / 1_000_000.0 * 5.00 + 500.0 / 1_000_000.0 * 15.00;
        assert!(
            (unknown_cost - fallback_expected).abs() < 1e-12,
            "unknown model cost {unknown_cost} should equal fallback {fallback_expected}"
        );
    }

    #[test]
    fn test_input_cost_per_million_known_models() {
        let models = [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
        ];
        for model in models {
            let cost = input_cost_per_million(model);
            assert!(
                cost > 0.0,
                "input cost for {model} should be positive, got {cost}"
            );
        }
    }

    #[test]
    fn test_output_cost_per_million_known_models() {
        let models = [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
        ];
        for model in models {
            let cost = output_cost_per_million(model);
            assert!(
                cost > 0.0,
                "output cost for {model} should be positive, got {cost}"
            );
        }
    }

    #[test]
    fn test_calculate_cost_zero_tokens() {
        let cost = calculate_cost("gpt-4o", 0, 0);
        assert_eq!(cost, 0.0, "zero tokens should have zero cost");
    }
}
