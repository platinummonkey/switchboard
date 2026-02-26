//! Semantic classifier trait and associated classification types.

use async_trait::async_trait;
use switchboard_common::types::{Message, Role, Tool};

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

// ── Token estimation ──────────────────────────────────────────────────────────

/// Estimate the number of tokens in a message list.
///
/// Uses a simple heuristic: `len(text) / 4 + 1` per message.  This avoids a
/// dependency on a full tokeniser while being accurate enough for complexity
/// classification.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| m.content.as_text().len() / 4 + 1)
        .sum()
}

// ── HeuristicClassifier ───────────────────────────────────────────────────────

/// A fast, purely heuristic [`SemanticClassifier`] that does not call any
/// external service.
///
/// All detection is synchronous keyword/pattern matching that completes in
/// well under 1 ms even for large prompts.
///
/// # Task type detection (in priority order)
///
/// 1. Non-empty `tools` list → [`TaskType::ToolUse`]
/// 2. Any message content contains triple backticks → [`TaskType::CodeGeneration`]
///    (language hint extracted from the fence opener if present)
/// 3. System prompt contains "review", "audit", or "analyze code" → [`TaskType::CodeReview`]
/// 4. System prompt contains "data", "csv", "sql", or "analytics" → [`TaskType::DataAnalysis`]
/// 5. System prompt contains "creative", "story", "poem", or "write" → [`TaskType::CreativeWriting`]
/// 6. Any message contains "step by step", "reason", "think through", or
///    "chain of thought" → [`TaskType::Reasoning`]
/// 7. Otherwise → [`TaskType::NaturalLanguage`]
///
/// # Complexity
///
/// | Token range      | Complexity |
/// |------------------|------------|
/// | < 500            | Simple     |
/// | 500 – 1999       | Medium     |
/// | ≥ 2000           | Complex    |
///
/// Override rules:
/// - Message count > 10 → bump to at least `Medium`
/// - `CodeGeneration` + `Simple` → bump to `Medium`
///
/// # Model recommendations
///
/// Always returned with confidence `0.8`.
pub struct HeuristicClassifier;

impl HeuristicClassifier {
    /// Create a new [`HeuristicClassifier`].
    pub fn new() -> Self {
        Self
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn detect_task_type(messages: &[Message], tools: &Option<Vec<Tool>>) -> TaskType {
        // 1. Tool use: non-empty tools list.
        if let Some(tools) = tools {
            if !tools.is_empty() {
                return TaskType::ToolUse;
            }
        }

        // 2. Code generation: triple backtick in any message.
        for msg in messages {
            let text = msg.content.as_text();
            if let Some(lang) = Self::extract_code_fence_language(&text) {
                return TaskType::CodeGeneration { language: lang };
            }
        }

        // Gather system prompt text for keyword checks 3-5.
        let system_text: String = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.as_text().to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        // 3. Code review.
        if system_text.contains("review")
            || system_text.contains("audit")
            || system_text.contains("analyze code")
        {
            return TaskType::CodeReview;
        }

        // 4. Data analysis.
        if system_text.contains("data")
            || system_text.contains("csv")
            || system_text.contains("sql")
            || system_text.contains("analytics")
        {
            return TaskType::DataAnalysis;
        }

        // 5. Creative writing.
        if system_text.contains("creative")
            || system_text.contains("story")
            || system_text.contains("poem")
            || system_text.contains("write")
        {
            return TaskType::CreativeWriting;
        }

        // 6. Reasoning keywords in any message.
        for msg in messages {
            let lower = msg.content.as_text().to_ascii_lowercase();
            if lower.contains("step by step")
                || lower.contains("reason")
                || lower.contains("think through")
                || lower.contains("chain of thought")
            {
                return TaskType::Reasoning;
            }
        }

        // 7. Default.
        TaskType::NaturalLanguage
    }

    /// Look for a code fence (` ``` `) in `text`.  Returns `Some(language)` if
    /// found, where `language` may be `None` when no language hint is given.
    fn extract_code_fence_language(text: &str) -> Option<Option<String>> {
        // Find the first occurrence of "```".
        let idx = text.find("```")?;
        // Extract the remainder of the line after "```".
        let rest = &text[idx + 3..];
        let line_end = rest.find('\n').unwrap_or(rest.len());
        let hint = rest[..line_end].trim().to_ascii_lowercase();
        if hint.is_empty() {
            Some(None)
        } else {
            Some(Some(hint))
        }
    }

    fn detect_complexity(
        estimated_tokens: usize,
        message_count: usize,
        task_type: &TaskType,
    ) -> Complexity {
        let mut complexity = if estimated_tokens < 500 {
            Complexity::Simple
        } else if estimated_tokens < 2000 {
            Complexity::Medium
        } else {
            Complexity::Complex
        };

        // Override: many messages → at least Medium.
        if message_count > 10 && complexity == Complexity::Simple {
            complexity = Complexity::Medium;
        }

        // Override: CodeGeneration + Simple → Medium.
        if matches!(task_type, TaskType::CodeGeneration { .. }) && complexity == Complexity::Simple
        {
            complexity = Complexity::Medium;
        }

        complexity
    }

    fn recommended_models(
        task_type: &TaskType,
        complexity: &Complexity,
    ) -> Vec<ModelRecommendation> {
        let models: &[&str] = match (task_type, complexity) {
            (TaskType::ToolUse, _) => &["claude-sonnet-4-20250514"],
            (TaskType::CodeGeneration { .. }, Complexity::Complex) => {
                &["claude-opus-4-20250514", "claude-sonnet-4-20250514"]
            }
            (TaskType::CodeGeneration { .. }, _) => {
                &["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"]
            }
            (TaskType::NaturalLanguage, Complexity::Simple) => &["claude-haiku-4-5-20251001"],
            _ => &["claude-sonnet-4-20250514"],
        };

        models
            .iter()
            .map(|&m| ModelRecommendation {
                model: m.to_string(),
                confidence: 0.8,
            })
            .collect()
    }
}

impl Default for HeuristicClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SemanticClassifier for HeuristicClassifier {
    async fn classify(&self, input: &ClassificationInput) -> ClassificationResult {
        let task_type = Self::detect_task_type(&input.messages, &input.tools);
        let complexity =
            Self::detect_complexity(input.estimated_tokens, input.messages.len(), &task_type);
        let recommended_models = Self::recommended_models(&task_type, &complexity);

        tracing::trace!(
            task_type = task_type.as_str(),
            complexity = complexity.as_str(),
            estimated_tokens = input.estimated_tokens,
            "heuristic classifier result"
        );

        ClassificationResult {
            task_type,
            complexity,
            recommended_models,
        }
    }
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

    // ── Helpers shared by HeuristicClassifier tests ───────────────────────────

    fn system_msg(text: &str) -> Message {
        Message {
            role: Role::System,
            content: MessageContent::Text(text.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    fn make_input(messages: Vec<Message>, estimated_tokens: usize) -> ClassificationInput {
        ClassificationInput {
            messages,
            tools: None,
            estimated_tokens,
        }
    }

    fn make_input_with_tools(
        messages: Vec<Message>,
        estimated_tokens: usize,
        tools: Vec<Tool>,
    ) -> ClassificationInput {
        ClassificationInput {
            messages,
            tools: Some(tools),
            estimated_tokens,
        }
    }

    // ── HeuristicClassifier tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_classify_tool_use() {
        use switchboard_common::types::Tool;
        let c = HeuristicClassifier::new();
        let tool = Tool {
            name: "get_weather".into(),
            description: Some("Get weather".into()),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let input = make_input_with_tools(vec![user_msg("What's the weather?")], 10, vec![tool]);
        let result = c.classify(&input).await;
        assert_eq!(result.task_type, TaskType::ToolUse);
    }

    #[tokio::test]
    async fn test_classify_code_generation_with_backticks() {
        let c = HeuristicClassifier::new();
        let input = make_input(
            vec![user_msg("Here is some code:\n```\nfn main() {}\n```")],
            20,
        );
        let result = c.classify(&input).await;
        assert!(
            matches!(result.task_type, TaskType::CodeGeneration { .. }),
            "expected CodeGeneration, got {:?}",
            result.task_type
        );
    }

    #[tokio::test]
    async fn test_classify_code_generation_detects_language() {
        let c = HeuristicClassifier::new();
        let input = make_input(
            vec![user_msg("Here is Rust:\n```rust\nfn main() {}\n```")],
            20,
        );
        let result = c.classify(&input).await;
        assert_eq!(
            result.task_type,
            TaskType::CodeGeneration {
                language: Some("rust".into())
            }
        );
    }

    #[tokio::test]
    async fn test_classify_natural_language_simple() {
        let c = HeuristicClassifier::new();
        let input = make_input(vec![user_msg("What is the capital of France?")], 10);
        let result = c.classify(&input).await;
        assert_eq!(result.task_type, TaskType::NaturalLanguage);
        assert_eq!(result.complexity, Complexity::Simple);
        // NaturalLanguage + Simple → haiku
        assert_eq!(
            result.recommended_models[0].model,
            "claude-haiku-4-5-20251001"
        );
    }

    #[tokio::test]
    async fn test_classify_reasoning_keywords() {
        let c = HeuristicClassifier::new();
        let input = make_input(
            vec![user_msg("Please think through this step by step.")],
            10,
        );
        let result = c.classify(&input).await;
        assert_eq!(result.task_type, TaskType::Reasoning);
    }

    #[tokio::test]
    async fn test_classify_data_analysis() {
        let c = HeuristicClassifier::new();
        let input = make_input(
            vec![
                system_msg("You are a data analytics assistant."),
                user_msg("Help me."),
            ],
            10,
        );
        let result = c.classify(&input).await;
        assert_eq!(result.task_type, TaskType::DataAnalysis);
    }

    #[tokio::test]
    async fn test_classify_complexity_simple() {
        let c = HeuristicClassifier::new();
        // 50 tokens → Simple
        let input = make_input(vec![user_msg("hi")], 50);
        let result = c.classify(&input).await;
        // NaturalLanguage + Simple (no bumping)
        assert_eq!(result.complexity, Complexity::Simple);
    }

    #[tokio::test]
    async fn test_classify_complexity_medium_by_tokens() {
        let c = HeuristicClassifier::new();
        // 1000 tokens → Medium
        let input = make_input(vec![user_msg("hi")], 1000);
        let result = c.classify(&input).await;
        assert_eq!(result.complexity, Complexity::Medium);
    }

    #[tokio::test]
    async fn test_classify_complexity_complex() {
        let c = HeuristicClassifier::new();
        // 3000 tokens → Complex
        let input = make_input(vec![user_msg("hi")], 3000);
        let result = c.classify(&input).await;
        assert_eq!(result.complexity, Complexity::Complex);
    }

    #[tokio::test]
    async fn test_classify_complexity_bumped_for_many_messages() {
        let c = HeuristicClassifier::new();
        // 11 messages with very few tokens each → bumped from Simple to Medium.
        let messages: Vec<Message> = (0..11).map(|_| user_msg("hi")).collect();
        // Use a small token count so complexity would be Simple without the bump.
        let input = make_input(messages, 30);
        let result = c.classify(&input).await;
        assert_eq!(result.complexity, Complexity::Medium);
    }

    #[test]
    fn test_estimate_tokens() {
        let messages = vec![
            user_msg("hello world"), // 11 chars / 4 + 1 = 3
            user_msg("foo"),         //  3 chars / 4 + 1 = 1
        ];
        // 3 + 1 = 4
        assert_eq!(estimate_tokens(&messages), 4);
    }

    // ── Property tests ────────────────────────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        /// The classifier always returns a valid TaskType for arbitrary messages.
        #[test]
        fn prop_classifier_always_returns_valid_task_type(
            texts in proptest::collection::vec("[a-zA-Z0-9 ]{0,200}", 1..=10usize),
        ) {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let result = rt.block_on(async {
                let c = HeuristicClassifier::new();
                let messages: Vec<Message> = texts
                    .iter()
                    .map(|t| user_msg(t))
                    .collect();
                let tokens = estimate_tokens(&messages);
                let input = ClassificationInput {
                    messages,
                    tools: None,
                    estimated_tokens: tokens,
                };
                c.classify(&input).await
            });
            // task_type.as_str() must return a non-empty string (valid variant)
            prop_assert!(!result.task_type.as_str().is_empty());
        }

        /// The classifier always returns a valid Complexity for arbitrary token counts.
        #[test]
        fn prop_classifier_always_returns_valid_complexity(
            token_count in 0usize..=100_000usize,
        ) {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let result = rt.block_on(async {
                let c = HeuristicClassifier::new();
                let input = ClassificationInput {
                    messages: vec![user_msg("test")],
                    tools: None,
                    estimated_tokens: token_count,
                };
                c.classify(&input).await
            });
            prop_assert!(
                result.complexity == Complexity::Simple
                    || result.complexity == Complexity::Medium
                    || result.complexity == Complexity::Complex,
                "unexpected complexity: {:?}",
                result.complexity
            );
        }
    }
}
