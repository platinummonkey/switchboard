pub mod health;
pub mod matcher;
pub mod selector;
pub mod semantic;

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use std::pin::Pin;
use switchboard_common::errors::SwitchboardError;
use switchboard_common::types::{ProxiedRequest, ProxiedResponse};

use crate::key_pool::PooledKey;

pub use health::ProviderHealthChecker;
pub use matcher::match_routing_rules;
pub use selector::ModelSelector;
pub use semantic::{
    ClassificationInput, ClassificationResult, Complexity, HeuristicClassifier,
    ModelRecommendation, SemanticClassifier, TaskType,
};

// ── Routing decision ──────────────────────────────────────────────────────────

/// Why a particular model was chosen for a request.
#[derive(Debug, Clone)]
pub enum SelectionReason {
    /// `[model_selection] mode = "static"`.
    Static,
    /// Matched a `[model_selection.mappings]` entry.
    Mapping,
    /// Client provided `X-Switchboard-Model` header.
    HeaderOverride,
    /// Per-user or per-team model override from admin API.
    UserOverride,
    /// Semantic classifier selected a model based on prompt analysis.
    Semantic {
        task_type: String,
        complexity: String,
    },
    /// No other rule matched; fell through to `fallback`.
    Fallback,
}

impl std::fmt::Display for SelectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionReason::Static => write!(f, "static"),
            SelectionReason::Mapping => write!(f, "mapping"),
            SelectionReason::HeaderOverride => write!(f, "header_override"),
            SelectionReason::UserOverride => write!(f, "user_override"),
            SelectionReason::Semantic {
                task_type,
                complexity,
            } => {
                write!(f, "semantic:{task_type}:{complexity}")
            }
            SelectionReason::Fallback => write!(f, "fallback"),
        }
    }
}

/// The resolved routing target for a single request.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Short provider name, e.g. `"anthropic"`.
    pub provider: String,
    /// The model ID to send to the upstream provider.
    pub model: String,
    /// Why this model was chosen (for observability).
    pub selection_reason: SelectionReason,
}

// ── Upstream provider trait ───────────────────────────────────────────────────

/// Type alias for a boxed streaming response body.
pub type BoxStream = Pin<Box<dyn Stream<Item = Result<Bytes, SwitchboardError>> + Send + 'static>>;

/// Abstraction over a single upstream LLM provider (Anthropic, OpenAI, etc.).
///
/// Implementations handle wire-format translation, request signing, and
/// response normalisation.  The proxy layer never touches provider-specific
/// formats directly.
///
/// Built-in implementations: `OpenAiProvider`, `AnthropicProvider`,
/// `BedrockProvider`, `VertexProvider`, `OllamaProvider` (Phases 7–8).
#[async_trait]
pub trait UpstreamProvider: Send + Sync {
    /// Short name used in metrics and config, e.g. `"anthropic"`.
    fn name(&self) -> &str;

    /// Wire format this provider uses, e.g. `"openai"` or `"anthropic"`.
    fn api_format(&self) -> &str;

    /// Returns `true` if this provider can serve the given model name.
    fn supports_model(&self, model: &str) -> bool;

    /// Send a non-streaming request and return the full response.
    async fn send(
        &self,
        request: ProxiedRequest,
        key: &PooledKey,
    ) -> Result<ProxiedResponse, SwitchboardError>;

    /// Send a streaming request and return a byte stream of SSE chunks.
    async fn send_streaming(
        &self,
        request: ProxiedRequest,
        key: &PooledKey,
    ) -> Result<BoxStream, SwitchboardError>;

    /// Probe the provider's availability. Returns `true` if reachable.
    async fn health_check(&self, key: &PooledKey) -> bool;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `UpstreamProvider` is object-safe.
    fn _assert_object_safe(_: &dyn UpstreamProvider) {}

    #[test]
    fn test_selection_reason_display() {
        assert_eq!(SelectionReason::Static.to_string(), "static");
        assert_eq!(SelectionReason::Mapping.to_string(), "mapping");
        assert_eq!(
            SelectionReason::HeaderOverride.to_string(),
            "header_override"
        );
        assert_eq!(SelectionReason::Fallback.to_string(), "fallback");
        assert_eq!(
            SelectionReason::Semantic {
                task_type: "code_generation".into(),
                complexity: "complex".into(),
            }
            .to_string(),
            "semantic:code_generation:complex"
        );
    }

    #[test]
    fn test_routing_decision_fields() {
        let d = RoutingDecision {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-20250514".into(),
            selection_reason: SelectionReason::Fallback,
        };
        assert_eq!(d.provider, "anthropic");
        assert_eq!(d.model, "claude-sonnet-4-20250514");
    }
}
