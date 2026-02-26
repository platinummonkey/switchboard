//! Integration tests for semantic routing: classification + rule matching +
//! model selector.

use std::collections::HashMap;

use switchboard_common::types::RequestContext;
use switchboard_common::types::{Message, MessageContent, Role, Tool};
use switchboard_server::config::model_selection::ModelSelectionConfig;
use switchboard_server::config::routing::{DefaultRouting, RoutingRule, SemanticRoutingConfig};
use switchboard_server::routing::selector::ModelSelector;
use switchboard_server::routing::{
    ClassificationInput, Complexity, HeuristicClassifier, SemanticClassifier, TaskType,
    match_routing_rules,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn user_msg(text: &str) -> Message {
    Message {
        role: Role::User,
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

fn make_tool(name: &str) -> Tool {
    Tool {
        name: name.into(),
        description: Some(format!("A tool named {name}")),
        input_schema: serde_json::json!({"type": "object", "properties": {}}),
    }
}

fn make_routing_config(
    rules: Vec<RoutingRule>,
    default_models: Vec<String>,
) -> SemanticRoutingConfig {
    SemanticRoutingConfig {
        enabled: true,
        classifier: "heuristic".into(),
        rules,
        default: DefaultRouting {
            preferred_models: default_models,
        },
    }
}

// ── HeuristicClassifier tests ─────────────────────────────────────────────────

/// 1. Message with a Python code fence → CodeGeneration { language: Some("python") }.
#[tokio::test]
async fn test_classify_code_generation_from_backticks() {
    let c = HeuristicClassifier::new();
    let input = make_input(
        vec![user_msg(
            "Here is my code:\n```python\nprint(\"hello\")\n```",
        )],
        20,
    );
    let result = c.classify(&input).await;
    assert_eq!(
        result.task_type,
        TaskType::CodeGeneration {
            language: Some("python".into())
        },
        "Python code fence should produce CodeGeneration {{ language: Some(\"python\") }}"
    );
}

/// 2. Message with code backticks AND a tools array → ToolUse takes priority.
#[tokio::test]
async fn test_classify_tool_use_takes_priority() {
    let c = HeuristicClassifier::new();
    // The message has a code fence but tools are also present.
    let input = make_input_with_tools(
        vec![user_msg(
            "Run the following:\n```python\nprint(\"hi\")\n```",
        )],
        20,
        vec![make_tool("execute_code")],
    );
    let result = c.classify(&input).await;
    assert_eq!(
        result.task_type,
        TaskType::ToolUse,
        "tools list must take priority over code fence detection"
    );
}

/// 3. Short user message with no special content → NaturalLanguage.
#[tokio::test]
async fn test_classify_simple_question_is_natural_language() {
    let c = HeuristicClassifier::new();
    let input = make_input(vec![user_msg("What is the capital of France?")], 10);
    let result = c.classify(&input).await;
    assert_eq!(
        result.task_type,
        TaskType::NaturalLanguage,
        "short question with no special keywords should be NaturalLanguage"
    );
}

/// 4. Single ~20-word message → Complexity::Simple.
#[tokio::test]
async fn test_complexity_simple_for_short_message() {
    let c = HeuristicClassifier::new();
    // Provide estimated_tokens < 500 to guarantee Simple.
    let msg = "Please summarise the history of the Roman Empire in one paragraph.";
    let input = make_input(vec![user_msg(msg)], 20);
    let result = c.classify(&input).await;
    // NaturalLanguage + Simple: no bumping rules apply.
    assert_eq!(
        result.complexity,
        Complexity::Simple,
        "~20-word message should be Simple complexity"
    );
}

/// 5. 15 messages averaging 200 words each (>> 2000 estimated tokens) →
///    Complexity::Complex.
#[tokio::test]
async fn test_complexity_complex_for_long_conversation() {
    let c = HeuristicClassifier::new();
    // Each "word" is roughly 5 chars; 200 words × 5 chars = 1000 chars → ~250
    // tokens per message.  15 messages → ~3750 tokens > 2000 → Complex.
    let long_text = "word ".repeat(200);
    let messages: Vec<Message> = (0..15).map(|_| user_msg(&long_text)).collect();
    // Compute a realistic token estimate: 15 * (1000/4 + 1) = 15 * 251 = 3765
    let estimated_tokens: usize = messages
        .iter()
        .map(|m| m.content.as_text().len() / 4 + 1)
        .sum();
    assert!(
        estimated_tokens >= 2000,
        "estimated_tokens={estimated_tokens} should be >= 2000"
    );
    let input = ClassificationInput {
        messages,
        tools: None,
        estimated_tokens,
    };
    let result = c.classify(&input).await;
    assert_eq!(
        result.complexity,
        Complexity::Complex,
        "long conversation should be Complex"
    );
}

/// 6. CodeGeneration + Complex → first recommended model is claude-opus-4-20250514.
#[tokio::test]
async fn test_recommended_models_for_complex_code() {
    let c = HeuristicClassifier::new();
    // Build a long code message to trigger Complex.
    let code_msg = format!("```rust\n{}\n```", "fn placeholder() {} ".repeat(500));
    let estimated_tokens = code_msg.len() / 4 + 1;
    assert!(
        estimated_tokens >= 2000,
        "must be large enough to be Complex"
    );
    let input = make_input(vec![user_msg(&code_msg)], estimated_tokens);
    let result = c.classify(&input).await;

    assert!(
        matches!(result.task_type, TaskType::CodeGeneration { .. }),
        "code fence → CodeGeneration"
    );
    assert_eq!(result.complexity, Complexity::Complex);
    assert!(
        !result.recommended_models.is_empty(),
        "must have at least one recommendation"
    );
    assert_eq!(
        result.recommended_models[0].model, "claude-opus-4-20250514",
        "CodeGeneration + Complex → first recommendation must be claude-opus-4-20250514"
    );
}

/// 7. NaturalLanguage + Simple → recommended models include claude-haiku-4-5-20251001.
#[tokio::test]
async fn test_recommended_models_for_simple_nl() {
    let c = HeuristicClassifier::new();
    let input = make_input(vec![user_msg("What time is it?")], 10);
    let result = c.classify(&input).await;

    assert_eq!(result.task_type, TaskType::NaturalLanguage);
    assert_eq!(result.complexity, Complexity::Simple);
    assert!(
        result
            .recommended_models
            .iter()
            .any(|r| r.model == "claude-haiku-4-5-20251001"),
        "NaturalLanguage + Simple should recommend claude-haiku-4-5-20251001"
    );
}

// ── Rule matching tests ───────────────────────────────────────────────────────

/// 8. Rule matching CodeGeneration + Complex → preferred models returned.
#[test]
fn test_routing_rules_matched_task_and_complexity() {
    use switchboard_server::routing::{ClassificationResult, ModelRecommendation};

    let config = make_routing_config(
        vec![RoutingRule {
            task_type: "code_generation".into(),
            complexity: Some("complex".into()),
            preferred_models: vec!["claude-opus-4-20250514".into()],
            fallback_models: vec![],
        }],
        vec!["claude-sonnet-4-20250514".into()],
    );

    let result = ClassificationResult {
        task_type: TaskType::CodeGeneration { language: None },
        complexity: Complexity::Complex,
        recommended_models: vec![ModelRecommendation {
            model: "claude-sonnet-4-20250514".into(),
            confidence: 0.8,
        }],
    };

    let models = match_routing_rules(&result, &config);
    assert_eq!(
        models,
        vec!["claude-opus-4-20250514"],
        "matching rule should return its preferred_models"
    );
}

/// 9. No rule matches → falls back to config.default.preferred_models.
#[test]
fn test_routing_rules_falls_back_to_default() {
    use switchboard_server::routing::{ClassificationResult, ModelRecommendation};

    let config = make_routing_config(
        vec![RoutingRule {
            task_type: "code_generation".into(),
            complexity: Some("complex".into()),
            preferred_models: vec!["claude-opus-4-20250514".into()],
            fallback_models: vec![],
        }],
        vec!["claude-haiku-4-5-20251001".into()],
    );

    // NaturalLanguage does not match the code_generation rule.
    let result = ClassificationResult {
        task_type: TaskType::NaturalLanguage,
        complexity: Complexity::Simple,
        recommended_models: vec![ModelRecommendation {
            model: "claude-haiku-4-5-20251001".into(),
            confidence: 0.8,
        }],
    };

    let models = match_routing_rules(&result, &config);
    assert_eq!(
        models,
        vec!["claude-haiku-4-5-20251001"],
        "no rule match should fall back to default preferred_models"
    );
}

/// 10. Rule has task_type = "tool_use" but no complexity → matches any complexity.
#[test]
fn test_routing_rule_without_complexity_matches_any() {
    use switchboard_server::routing::{ClassificationResult, ModelRecommendation};

    let config = make_routing_config(
        vec![RoutingRule {
            task_type: "tool_use".into(),
            complexity: None, // wildcard: matches any complexity
            preferred_models: vec!["claude-sonnet-4-20250514".into()],
            fallback_models: vec![],
        }],
        vec!["claude-haiku-4-5-20251001".into()],
    );

    let result = ClassificationResult {
        task_type: TaskType::ToolUse,
        complexity: Complexity::Complex,
        recommended_models: vec![ModelRecommendation {
            model: "claude-sonnet-4-20250514".into(),
            confidence: 0.8,
        }],
    };

    let models = match_routing_rules(&result, &config);
    assert_eq!(
        models,
        vec!["claude-sonnet-4-20250514"],
        "rule with no complexity filter should match ToolUse + Complex"
    );
}

// ── ModelSelector integration tests ──────────────────────────────────────────

/// 11. Semantic mode returns (fallback_model, SelectionReason::Fallback) synchronously.
#[test]
fn test_model_selector_semantic_mode_returns_fallback_sync() {
    use switchboard_server::routing::SelectionReason;

    let config = ModelSelectionConfig {
        mode: "semantic".into(),
        fallback: Some("claude-sonnet-4-20250514".into()),
        ..ModelSelectionConfig::default()
    };

    let selector = ModelSelector::new(config);
    let ctx = RequestContext::new();
    let (model, reason) = selector.select(&ctx);

    assert_eq!(
        model, "claude-sonnet-4-20250514",
        "semantic mode must return the configured fallback synchronously"
    );
    assert!(
        matches!(reason, SelectionReason::Fallback),
        "reason must be Fallback for semantic mode synchronous path"
    );
}

/// 12. Dynamic mode with per-user override → returns (override_model, UserOverride).
#[test]
fn test_model_selector_dynamic_mode_with_override() {
    use switchboard_server::routing::SelectionReason;

    let mut overrides = HashMap::new();
    overrides.insert(
        "alice@example.com".to_string(),
        "claude-opus-4-20250514".to_string(),
    );

    let config = ModelSelectionConfig {
        mode: "dynamic".into(),
        header: "x-switchboard-model".into(),
        fallback: Some("claude-sonnet-4-20250514".into()),
        overrides,
        ..ModelSelectionConfig::default()
    };

    let selector = ModelSelector::new(config);
    let ctx = RequestContext {
        user_id: Some("alice@example.com".into()),
        ..RequestContext::new()
    };

    let (model, reason) = selector.select(&ctx);

    assert_eq!(
        model, "claude-opus-4-20250514",
        "user override should return the configured override model"
    );
    assert!(
        matches!(reason, SelectionReason::UserOverride),
        "reason must be UserOverride when a per-user override matches"
    );
}
