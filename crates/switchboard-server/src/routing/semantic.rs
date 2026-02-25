//! Semantic classifier trait and associated classification types.

use async_trait::async_trait;
use switchboard_common::types::{Message, Tool};

// ── Task type ─────────────────────────────────────────────────────────────────

/// Detected category of the incoming prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskType {
    CodeGeneration { language: Option<String> },
    CodeReview,
    NaturalLanguage,
    Reasoning,
    CreativeWriting,
    DataAnalysis,
    ToolUse,
    Unknown,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::CodeGeneration { .. } => "code_generation",
            TaskType::CodeReview => "code_review",
            TaskType::NaturalLanguage => "natural_language",
            TaskType::Reasoning => "reasoning",
            TaskType::CreativeWriting => "creative_writing",
            TaskType::DataAnalysis => "data_analysis",
            TaskType::ToolUse => "tool_use",
            TaskType::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::CodeGeneration {
                language: Some(lang),
            } => {
                write!(f, "code_generation:{lang}")
            }
            other => write!(f, "{}", other.as_str()),
        }
    }
}

// ── Complexity ────────────────────────────────────────────────────────────────

/// Estimated complexity of the task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Complexity {
    /// Short lookups, simple formatting, single-step answers.
    Simple,
    /// Standard code tasks, moderate multi-step reasoning.
    Medium,
    /// Architecture design, long generation, deep reasoning chains.
    Complex,
}

impl Complexity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Complexity::Simple => "simple",
            Complexity::Medium => "medium",
            Complexity::Complex => "complex",
        }
    }
}

impl std::fmt::Display for Complexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Model recommendation ──────────────────────────────────────────────────────

/// A model suggested by the semantic classifier for a given task.
#[derive(Debug, Clone)]
pub struct ModelRecommendation {
    pub model: String,
    /// Classifier's confidence in this recommendation (0.0–1.0).
    pub confidence: f64,
}

// ── Classification input / result ─────────────────────────────────────────────

/// Everything the classifier needs to produce a routing decision.
#[derive(Debug, Clone)]
pub struct ClassificationInput {
    pub messages: Vec<Message>,
    pub tools: Option<Vec<Tool>>,
    /// Pre-computed token count estimate (avoids double-counting).
    pub estimated_tokens: usize,
}

/// The classifier's verdict for a request.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub task_type: TaskType,
    pub complexity: Complexity,
    /// Ordered list of recommended models (most preferred first).
    pub recommended_models: Vec<ModelRecommendation>,
}

// ── SemanticClassifier trait ──────────────────────────────────────────────────

/// Classifies a prompt into a task type and complexity without calling an LLM.
///
/// The classification runs on every request in the hot path, so it must be
/// fast (target: < 1 ms).
///
/// Built-in implementation: `HeuristicClassifier` (Phase 11b).
#[async_trait]
pub trait SemanticClassifier: Send + Sync {
    /// Classify the prompt. Must not block or perform I/O.
    async fn classify(&self, input: &ClassificationInput) -> ClassificationResult;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use switchboard_common::types::{MessageContent, Role};

    use super::*;

    /// Verify `SemanticClassifier` is object-safe.
    fn _assert_object_safe(_: &dyn SemanticClassifier) {}

    /// A stub classifier that always says "unknown / simple".
    struct TrivialClassifier;

    #[async_trait]
    impl SemanticClassifier for TrivialClassifier {
        async fn classify(&self, _input: &ClassificationInput) -> ClassificationResult {
            ClassificationResult {
                task_type: TaskType::Unknown,
                complexity: Complexity::Simple,
                recommended_models: vec![ModelRecommendation {
                    model: "claude-haiku-4-5-20251001".into(),
                    confidence: 0.5,
                }],
            }
        }
    }

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: MessageContent::Text(text.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    #[tokio::test]
    async fn test_trivial_classifier() {
        let c = TrivialClassifier;
        let input = ClassificationInput {
            messages: vec![user_msg("hello world")],
            tools: None,
            estimated_tokens: 3,
        };
        let result = c.classify(&input).await;
        assert_eq!(result.task_type, TaskType::Unknown);
        assert_eq!(result.complexity, Complexity::Simple);
        assert_eq!(result.recommended_models.len(), 1);
    }

    #[test]
    fn test_task_type_as_str() {
        assert_eq!(
            TaskType::CodeGeneration { language: None }.as_str(),
            "code_generation"
        );
        assert_eq!(TaskType::ToolUse.as_str(), "tool_use");
        assert_eq!(TaskType::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_task_type_display_with_language() {
        let t = TaskType::CodeGeneration {
            language: Some("rust".into()),
        };
        assert_eq!(t.to_string(), "code_generation:rust");
    }

    #[test]
    fn test_task_type_display_no_language() {
        let t = TaskType::CodeGeneration { language: None };
        assert_eq!(t.to_string(), "code_generation");
    }

    #[test]
    fn test_complexity_display() {
        assert_eq!(Complexity::Simple.to_string(), "simple");
        assert_eq!(Complexity::Medium.to_string(), "medium");
        assert_eq!(Complexity::Complex.to_string(), "complex");
    }
}
