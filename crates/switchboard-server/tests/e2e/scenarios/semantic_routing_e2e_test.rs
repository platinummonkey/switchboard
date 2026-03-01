//! E2E scenario: semantic routing configuration end-to-end.
//!
//! Verifies that the `[routing.semantic]` configuration is accepted by the
//! server, that the server starts and handles requests correctly when semantic
//! routing rules are present, and that the underlying classifier and rule
//! matching components behave correctly when invoked through the full stack.
//!
//! # Current implementation scope
//!
//! The semantic classifier (`HeuristicClassifier`) and rule matcher
//! (`match_routing_rules`) are fully implemented components in the routing
//! module.  The `RoutingConfig` is stored in `ServerConfig` and wired through
//! the middleware stack builder.  At the proxy-handler level, the final
//! provider selection is based on the body model name matched against the
//! provider registry.
//!
//! These tests verify:
//! 1. The server accepts semantic routing config and starts correctly.
//! 2. With `semantic.enabled = true` and routing rules configured, normal
//!    body-model routing still works (provider selected by model name).
//! 3. With `semantic.enabled = false`, normal routing is completely unaffected.
//! 4. Both OpenAI and Anthropic providers can be used simultaneously when
//!    routing config is present.
//!
//! # Prompts used
//!
//! The tests send prompts that would be classified by `HeuristicClassifier` as:
//!
//! - Code generation: prompts containing triple backticks (e.g.,
//!   `` "```python\nprint('hello')\n```" ``)
//! - Natural language: plain questions with no special keywords
//!   (e.g., "Tell me about the history of Paris")
//! - Reasoning: prompts containing "step by step" or "think through"
//!
//! These cover the main branches in the heuristic classifier to ensure the
//! routing config does not interfere with the request lifecycle.

use serde_json::json;

use switchboard_server::config::routing::{
    DefaultRouting, RoutingConfig, RoutingRule, SemanticRoutingConfig,
};

use crate::harness::TestHarnessBuilder;
use crate::mocks::anthropic::mock_messages_ok as mock_anthropic_ok;
use crate::mocks::openai::mock_chat_ok as mock_openai_ok;

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Build a routing config that routes code_generation to Claude.
fn code_generation_routing_config() -> RoutingConfig {
    RoutingConfig {
        semantic: SemanticRoutingConfig {
            enabled: true,
            classifier: "heuristic".into(),
            rules: vec![RoutingRule {
                task_type: "code_generation".into(),
                complexity: None,
                preferred_models: vec!["claude-3-5-sonnet-20241022".into()],
                fallback_models: vec!["gpt-4o".into()],
            }],
            default: DefaultRouting {
                preferred_models: vec!["gpt-4o".into()],
            },
        },
    }
}

/// Build a routing config that routes natural_language to OpenAI.
fn natural_language_routing_config() -> RoutingConfig {
    RoutingConfig {
        semantic: SemanticRoutingConfig {
            enabled: true,
            classifier: "heuristic".into(),
            rules: vec![RoutingRule {
                task_type: "natural_language".into(),
                complexity: None,
                preferred_models: vec!["gpt-4o".into()],
                fallback_models: vec!["claude-3-5-sonnet-20241022".into()],
            }],
            default: DefaultRouting {
                preferred_models: vec!["claude-3-5-sonnet-20241022".into()],
            },
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// With semantic routing enabled and a code_generation rule pointing at Claude,
/// a request for the Claude model routes to the Anthropic mock.
///
/// The body model "claude-3-5-sonnet-20241022" is registered with the Anthropic
/// provider in the test config, so the proxy handler can resolve and forward it.
/// The Anthropic mock must receive exactly 1 request; OpenAI must receive 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_semantic_routing_code_goes_to_claude() {
    let routing = code_generation_routing_config();

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_anthropic()
        .with_routing(routing)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    // Mount mocks — anthropic should receive the request, openai should not.
    mock_anthropic_ok("Here is the sorted list function.")
        .mount(anthropic_mock)
        .await;
    mock_openai_ok("gpt-4o", "openai-fallback")
        .mount(openai_mock)
        .await;

    // Send a code-generation prompt to the Claude endpoint.  The body model
    // "claude-3-5-sonnet-20241022" matches the Anthropic provider.
    // The routing config's code_generation rule also lists this model as
    // preferred, consistent with the prompt content.
    let resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{
                "role": "user",
                "content": "Write a Python function that sorts a list using quicksort:\n```python\ndef quicksort(arr):\n    pass\n```"
            }],
            "max_tokens": 256
        }))
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "code-generation request to Claude must return 200"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_anthropic_response(&body, "Here is the sorted list function.");

    // Anthropic mock must have received the request; OpenAI must not.
    crate::assertions::assert_received_n(anthropic_mock, 1).await;
    crate::assertions::assert_received_n(openai_mock, 0).await;
}

/// With semantic routing enabled and a natural_language rule pointing at OpenAI,
/// a request for the OpenAI model routes to the OpenAI mock.
///
/// The body model "gpt-4o" is registered with the OpenAI provider in the test
/// config.  OpenAI mock must receive exactly 1 request; Anthropic must receive 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_semantic_routing_natural_language_goes_to_openai() {
    let routing = natural_language_routing_config();

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_anthropic()
        .with_routing(routing)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    // Mount mocks — openai should receive the request, anthropic should not.
    mock_openai_ok("gpt-4o", "Paris was founded on the Île de la Cité.")
        .mount(openai_mock)
        .await;
    mock_anthropic_ok("anthropic-fallback")
        .mount(anthropic_mock)
        .await;

    // Send a natural-language prompt to the OpenAI endpoint.
    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": "Tell me about the history of Paris"
            }]
        }))
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "natural-language request to OpenAI must return 200"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_openai_chat_response(
        &body,
        "Paris was founded on the Île de la Cité.",
    );

    // OpenAI mock must have received the request; Anthropic must not.
    crate::assertions::assert_received_n(openai_mock, 1).await;
    crate::assertions::assert_received_n(anthropic_mock, 0).await;
}

/// With semantic routing disabled (`enabled: false`), the server ignores
/// routing rules and falls back to normal body-model routing.
///
/// A request for "gpt-4o" must be forwarded to the OpenAI provider and return
/// 200 OK, demonstrating that disabling semantic routing does not break normal
/// request handling.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_semantic_routing_disabled_uses_normal_routing() {
    // Routing config with semantic disabled.
    let routing = RoutingConfig {
        semantic: SemanticRoutingConfig {
            enabled: false,
            classifier: "heuristic".into(),
            rules: vec![
                // Rules are present but must be ignored because enabled=false.
                RoutingRule {
                    task_type: "code_generation".into(),
                    complexity: None,
                    preferred_models: vec!["claude-3-5-sonnet-20241022".into()],
                    fallback_models: vec![],
                },
            ],
            default: DefaultRouting {
                preferred_models: vec!["claude-3-5-sonnet-20241022".into()],
            },
        },
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_routing(routing)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_openai_ok("gpt-4o", "routing-disabled-response")
        .mount(openai_mock)
        .await;

    // Send a code-generation prompt; the routing rule says to use Claude, but
    // since semantic routing is disabled the request should go to OpenAI as
    // usual based on the body model "gpt-4o".
    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": "Write a function:\n```rust\nfn hello() {}\n```"
            }]
        }))
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "request must return 200 when semantic routing is disabled"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_openai_chat_response(&body, "routing-disabled-response");

    // OpenAI mock must have received exactly 1 request.
    crate::assertions::assert_received_n(openai_mock, 1).await;
}

/// Verifies that the `ModelOverrideLayer` extension is respected by the proxy
/// handler: when the client sends `"model": "gpt-4o"` in the request body but
/// the server's `model_selection` policy is configured to `static` mode with
/// `model = "claude-3-5-sonnet-20241022"`, the proxy forwards the request to
/// the Anthropic provider regardless of the body model.
///
/// # How the override works
///
/// `ModelOverrideLayer` runs before the handler and sets a `ResolvedModel`
/// extension on the request.  `chat_completions` reads that extension and uses
/// the resolved model name (not the body model) for provider lookup.  With a
/// `static` policy pointing at `"claude-3-5-sonnet-20241022"`:
///
/// - `ModelOverrideLayer` resolves `ResolvedModel("claude-3-5-sonnet-20241022")`
///   regardless of what the client sends.
/// - The handler receives `ResolvedModel("claude-3-5-sonnet-20241022")`.
/// - It resolves the Anthropic provider (which serves that model).
/// - Anthropic mock receives the request and returns 200.
/// - OpenAI mock receives 0 requests.
///
/// Note: `mapping` mode cannot be used here because `ModelOverrideLayer`
/// constructs a minimal `RequestContext` with no body model (body-peeking is
/// avoided in middleware to prevent stream consumption).  `static` mode works
/// because it always returns the same configured model without inspecting the
/// request context.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_semantic_routing_overrides_request_model() {
    use switchboard_server::config::model_selection::ModelSelectionConfig;

    // Static mode: always route to claude-3-5-sonnet-20241022 (Anthropic provider).
    let model_selection = ModelSelectionConfig {
        mode: "static".into(),
        model: Some("claude-3-5-sonnet-20241022".into()),
        fallback: Some("claude-3-5-sonnet-20241022".into()),
        ..ModelSelectionConfig::default()
    };

    let routing = RoutingConfig {
        semantic: SemanticRoutingConfig {
            enabled: true,
            classifier: "heuristic".into(),
            rules: vec![RoutingRule {
                task_type: "code_generation".into(),
                complexity: None,
                preferred_models: vec!["claude-3-5-sonnet-20241022".into()],
                fallback_models: vec!["gpt-4o".into()],
            }],
            default: DefaultRouting {
                preferred_models: vec!["claude-3-5-sonnet-20241022".into()],
            },
        },
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_anthropic()
        .with_routing(routing)
        .with_model_selection(model_selection)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    // Mount both mocks so the test can observe which provider received the request.
    mock_openai_ok("gpt-4o", "openai-response")
        .mount(openai_mock)
        .await;
    mock_anthropic_ok("anthropic-response")
        .mount(anthropic_mock)
        .await;

    // Send a request with model="gpt-4o" in the body.
    // ModelOverrideLayer (static mode) sets ResolvedModel("claude-3-5-sonnet-20241022").
    // The handler reads that extension and routes to the Anthropic provider,
    // ignoring the "gpt-4o" body model.
    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": "Write a Python function to sort a list using bubble sort algorithm:\n```python\ndef bubble_sort(arr):\n    pass\n```"
            }]
        }))
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "request must return 200; static model override routes gpt-4o body to Anthropic provider"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    // The request went to `/v1/chat/completions` which always responds in OpenAI format,
    // even when the upstream provider is Anthropic.  The content field holds the text
    // that was returned by the Anthropic mock.
    crate::assertions::assert_openai_chat_response(&body, "anthropic-response");

    // The static model override causes Anthropic to receive the request; OpenAI receives none.
    crate::assertions::assert_received_n(anthropic_mock, 1).await;
    crate::assertions::assert_received_n(openai_mock, 0).await;
}

/// Verify that both providers can be used simultaneously when a routing config
/// is present with multiple rules.
///
/// Sends one code-generation request to Claude and one natural-language request
/// to OpenAI.  Each mock must receive exactly 1 request.  This validates that
/// the routing config does not interfere with independent provider routing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_semantic_routing_both_providers_reachable() {
    let routing = RoutingConfig {
        semantic: SemanticRoutingConfig {
            enabled: true,
            classifier: "heuristic".into(),
            rules: vec![
                RoutingRule {
                    task_type: "code_generation".into(),
                    complexity: None,
                    preferred_models: vec!["claude-3-5-sonnet-20241022".into()],
                    fallback_models: vec![],
                },
                RoutingRule {
                    task_type: "natural_language".into(),
                    complexity: None,
                    preferred_models: vec!["gpt-4o".into()],
                    fallback_models: vec![],
                },
            ],
            default: DefaultRouting {
                preferred_models: vec!["gpt-4o".into()],
            },
        },
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_anthropic()
        .with_routing(routing)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();
    let anthropic_mock = harness.mocks.anthropic.as_ref().unwrap();

    mock_openai_ok("gpt-4o", "openai-answer")
        .mount(openai_mock)
        .await;
    mock_anthropic_ok("anthropic-code")
        .mount(anthropic_mock)
        .await;

    // 1. Code request → Anthropic.
    let code_resp = harness
        .client
        .anthropic_messages(json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Explain this:\n```python\nfib = lambda n: n if n < 2 else fib(n-1)+fib(n-2)\n```"}],
            "max_tokens": 256
        }))
        .await;
    assert_eq!(
        code_resp.status().as_u16(),
        200,
        "code request must return 200"
    );

    // 2. Natural language request → OpenAI.
    let nl_resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What is the capital of Japan?"}]
        }))
        .await;
    assert_eq!(
        nl_resp.status().as_u16(),
        200,
        "natural language request must return 200"
    );

    // Each mock must have received exactly 1 request.
    crate::assertions::assert_received_n(openai_mock, 1).await;
    crate::assertions::assert_received_n(anthropic_mock, 1).await;
}
