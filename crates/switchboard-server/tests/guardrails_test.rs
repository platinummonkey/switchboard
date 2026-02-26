//! Integration tests for the guardrail pipeline.
//!
//! Covers pipeline orchestration, fixture-driven scenarios, streaming modes,
//! and fail-open / fail-closed behaviour.

use std::collections::HashMap;

use switchboard_server::config::guardrails::{EngineConfig, GuardrailsConfig, RegexRule};
use switchboard_server::guardrails::engine::GuardrailInput;
use switchboard_server::guardrails::pipeline::{GuardrailPipeline, StreamingMode};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_input(content: &str) -> GuardrailInput {
    GuardrailInput {
        content: content.into(),
        messages: vec![],
        user: None,
        model: "claude-sonnet-4-20250514".into(),
        metadata: HashMap::new(),
    }
}

fn make_input_from_fixture(json: &str) -> GuardrailInput {
    let v: serde_json::Value = serde_json::from_str(json).expect("fixture is valid JSON");
    GuardrailInput {
        content: v["content"].as_str().unwrap_or("").into(),
        messages: vec![],
        user: None,
        model: v["model"]
            .as_str()
            .unwrap_or("claude-sonnet-4-20250514")
            .into(),
        metadata: HashMap::new(),
    }
}

// ── Pipeline orchestration tests ──────────────────────────────────────────────

/// 1. First engine blocks on `\btest\b`; second engine (keyword) should never
///    run because the pipeline is fail-fast.
#[tokio::test]
async fn test_pipeline_blocks_on_first_engine_match() {
    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "1s".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![
            EngineConfig {
                engine_type: "builtin_regex".into(),
                phase: "pre_request".into(),
                rules: vec![RegexRule {
                    name: "test_word".into(),
                    pattern: r"\btest\b".into(),
                    action: "block".into(),
                }],
                action: None,
                keywords: vec![],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
            EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "pre_request".into(),
                rules: vec![],
                action: Some("block".into()),
                // Use a keyword that should NOT appear in the input so if
                // this engine ran it would pass — we only want to verify
                // the pipeline stops at engine 1.
                keywords: vec!["never_present_xyz".into()],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
        ],
    };

    let pipeline = GuardrailPipeline::from_config(&cfg).expect("pipeline builds");
    let verdict = pipeline
        .evaluate_request(&make_input("this is a test input"))
        .await;

    assert!(
        verdict.action.is_blocking(),
        "regex engine should have blocked the input"
    );
    assert_eq!(
        verdict.engine, "regex",
        "first engine (regex) should have fired"
    );
}

/// 2. All engines pass; pipeline verdict is Pass.
#[tokio::test]
async fn test_pipeline_passes_when_all_engines_pass() {
    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "1s".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![
            EngineConfig {
                engine_type: "builtin_regex".into(),
                phase: "pre_request".into(),
                rules: vec![RegexRule {
                    name: "ssn_pattern".into(),
                    pattern: r"\b\d{3}-\d{2}-\d{4}\b".into(),
                    action: "block".into(),
                }],
                action: None,
                keywords: vec![],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
            EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "pre_request".into(),
                rules: vec![],
                action: Some("block".into()),
                keywords: vec!["forbidden_word".into()],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
        ],
    };

    let pipeline = GuardrailPipeline::from_config(&cfg).expect("pipeline builds");
    // Clean input — no SSN, no forbidden keyword.
    let verdict = pipeline
        .evaluate_request(&make_input("What is the capital of France?"))
        .await;

    assert!(
        verdict.action.is_pass(),
        "all engines pass → pipeline passes"
    );
}

/// 3. Pipeline with fail_mode = "open"; engine errors treated as Pass.
#[tokio::test]
async fn test_pipeline_fail_open_continues_on_engine_error() {
    // Use engine_from_config with type "grpc" which requires an endpoint and
    // will fail to construct synchronously (unknown type for sync factory).
    // Instead we use the async path indirectly: construct a pipeline via config
    // that has a "builtin_token_limit" with a valid max_input_tokens limit that
    // would block, then test a failing HTTP engine (no server) in open mode
    // via HttpCalloutEngine directly.
    //
    // Simplest approach that exercises fail-open on engine error: construct a
    // pipeline from config with `fail_mode = "open"` and an HTTP engine pointing
    // at a URL with no server. Because the HttpCalloutEngine returns an Err on
    // connection failure, the pipeline must treat it as Pass.
    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "100ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "http".into(),
            phase: "pre_request".into(),
            rules: vec![],
            action: None,
            keywords: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            // Point at a port that is almost certainly not listening.
            endpoint: Some("http://127.0.0.1:19999/evaluate".into()),
            timeout: Some("50ms".into()),
            tls: false,
            headers: Default::default(),
        }],
    };

    let pipeline = GuardrailPipeline::from_config(&cfg).expect("pipeline builds");
    // The HTTP engine will fail (connection refused) and fail-open should
    // produce a Pass verdict.
    let verdict = pipeline.evaluate_request(&make_input("some input")).await;
    assert!(
        verdict.action.is_pass(),
        "fail-open: connection error should be treated as pass, got {:?}",
        verdict.action
    );
}

/// 4. Pre-request and post-response engines are separate; calling
///    `evaluate_request` runs only pre-request engines.
#[tokio::test]
async fn test_pipeline_pre_and_post_separation() {
    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "1s".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![
            EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "pre_request".into(),
                rules: vec![],
                action: Some("block".into()),
                keywords: vec!["pre_keyword".into()],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
            EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "pre_request".into(),
                rules: vec![],
                action: Some("block".into()),
                keywords: vec!["second_pre_keyword".into()],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
            // Post-response engine: blocks on "post_keyword".
            EngineConfig {
                engine_type: "builtin_keyword".into(),
                phase: "post_response".into(),
                rules: vec![],
                action: Some("block".into()),
                keywords: vec!["post_keyword".into()],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
        ],
    };

    let pipeline = GuardrailPipeline::from_config(&cfg).expect("pipeline builds");

    // Input contains the post_keyword but NOT any pre keyword.
    // evaluate_request should pass (post engine does not run).
    let verdict = pipeline
        .evaluate_request(&make_input("this contains post_keyword"))
        .await;
    assert!(
        verdict.action.is_pass(),
        "post engine must not run during evaluate_request"
    );

    // evaluate_request on input with pre keyword should block.
    let verdict2 = pipeline
        .evaluate_request(&make_input("this contains pre_keyword"))
        .await;
    assert!(
        verdict2.action.is_blocking(),
        "pre engine must block when keyword matched"
    );
}

/// 5. `GuardrailPipeline::from_config` builds correctly from a config with one
///    pre-request regex engine and one post-response secret-detection engine.
#[tokio::test]
async fn test_pipeline_from_config_builds_correctly() {
    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![
            EngineConfig {
                engine_type: "builtin_regex".into(),
                phase: "pre_request".into(),
                rules: vec![RegexRule {
                    name: "ssn".into(),
                    pattern: r"\b\d{3}-\d{2}-\d{4}\b".into(),
                    action: "block".into(),
                }],
                action: None,
                keywords: vec![],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
            EngineConfig {
                engine_type: "builtin_secret_detection".into(),
                phase: "post_response".into(),
                rules: vec![],
                action: Some("block".into()),
                keywords: vec![],
                max_input_tokens: None,
                max_output_tokens: None,
                endpoint: None,
                timeout: None,
                tls: false,
                headers: Default::default(),
            },
        ],
    };

    // Must build without error.
    let result = GuardrailPipeline::from_config(&cfg);
    assert!(
        result.is_ok(),
        "pipeline should build from config without error"
    );
}

/// 6. Pipeline with a very short timeout and an HTTP engine (no server): in
///    fail-open mode the timeout results in a Pass verdict.
#[tokio::test]
async fn test_pipeline_timeout_returns_pass_in_open_mode() {
    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        // Very short pipeline timeout — the HTTP engine will time out.
        timeout: "10ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "http".into(),
            phase: "pre_request".into(),
            rules: vec![],
            action: None,
            keywords: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            // No server listening → connection will be refused; even if
            // timeout fires first, both paths produce Pass in open mode.
            endpoint: Some("http://127.0.0.1:19998/evaluate".into()),
            timeout: Some("5000ms".into()), // engine timeout > pipeline timeout
            tls: false,
            headers: Default::default(),
        }],
    };

    let pipeline = GuardrailPipeline::from_config(&cfg).expect("pipeline builds");
    let verdict = pipeline.evaluate_request(&make_input("hello")).await;
    assert!(
        verdict.action.is_pass(),
        "fail-open timeout/error should produce Pass, got {:?}",
        verdict.action
    );
}

// ── Fixture-driven tests ───────────────────────────────────────────────────────

/// 7. SSN prompt is blocked by a regex engine with the SSN pattern.
#[tokio::test]
async fn test_fixture_ssn_prompt_blocked_by_regex() {
    let fixture_json = include_str!("fixtures/guardrails/ssn_prompt.json");
    let input = make_input_from_fixture(fixture_json);

    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "builtin_regex".into(),
            phase: "pre_request".into(),
            rules: vec![RegexRule {
                name: "ssn".into(),
                pattern: r"\b\d{3}-\d{2}-\d{4}\b".into(),
                action: "block".into(),
            }],
            action: None,
            keywords: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            endpoint: None,
            timeout: None,
            tls: false,
            headers: Default::default(),
        }],
    };

    let pipeline = GuardrailPipeline::from_config(&cfg).expect("pipeline builds");
    let verdict = pipeline.evaluate_request(&input).await;
    assert!(
        verdict.action.is_blocking(),
        "SSN prompt should be blocked by regex engine"
    );
}

/// 8. Clean prompt passes with the same SSN regex config.
#[tokio::test]
async fn test_fixture_clean_prompt_passes() {
    let fixture_json = include_str!("fixtures/guardrails/clean_prompt.json");
    let input = make_input_from_fixture(fixture_json);

    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "builtin_regex".into(),
            phase: "pre_request".into(),
            rules: vec![RegexRule {
                name: "ssn".into(),
                pattern: r"\b\d{3}-\d{2}-\d{4}\b".into(),
                action: "block".into(),
            }],
            action: None,
            keywords: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            endpoint: None,
            timeout: None,
            tls: false,
            headers: Default::default(),
        }],
    };

    let pipeline = GuardrailPipeline::from_config(&cfg).expect("pipeline builds");
    let verdict = pipeline.evaluate_request(&input).await;
    assert!(
        verdict.action.is_pass(),
        "clean prompt should pass the SSN regex engine"
    );
}

/// 9. API key prompt is blocked by the secret detection engine.
#[tokio::test]
async fn test_fixture_api_key_detected() {
    let fixture_json = include_str!("fixtures/guardrails/api_key_prompt.json");
    let input = make_input_from_fixture(fixture_json);

    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "builtin_secret_detection".into(),
            phase: "pre_request".into(),
            rules: vec![],
            action: Some("block".into()),
            keywords: vec![],
            max_input_tokens: None,
            max_output_tokens: None,
            endpoint: None,
            timeout: None,
            tls: false,
            headers: Default::default(),
        }],
    };

    let pipeline = GuardrailPipeline::from_config(&cfg).expect("pipeline builds");
    let verdict = pipeline.evaluate_request(&input).await;
    assert!(
        verdict.action.is_blocking(),
        "API key prompt should be blocked by secret detection engine"
    );
}

// ── Streaming mode tests ──────────────────────────────────────────────────────

/// 10. `StreamingMode::AsyncAudit` variant exists and a pipeline with
///     `streaming_mode = "async_audit"` builds successfully.
#[tokio::test]
async fn test_streaming_mode_async_audit_defined() {
    // Verify the variant is accessible and has the expected value.
    let mode = StreamingMode::AsyncAudit;
    // Use it in an assertion so the compiler doesn't optimise it away.
    assert!(
        matches!(mode, StreamingMode::AsyncAudit),
        "StreamingMode::AsyncAudit must be accessible"
    );

    let cfg = GuardrailsConfig {
        enabled: true,
        fail_mode: "open".into(),
        timeout: "500ms".into(),
        streaming_mode: "async_audit".into(),
        engines: vec![EngineConfig {
            engine_type: "builtin_keyword".into(),
            phase: "pre_request".into(),
            rules: vec![],
            action: Some("block".into()),
            keywords: vec!["bad_word".into()],
            max_input_tokens: None,
            max_output_tokens: None,
            endpoint: None,
            timeout: None,
            tls: false,
            headers: Default::default(),
        }],
    };

    let result = GuardrailPipeline::from_config(&cfg);
    assert!(
        result.is_ok(),
        "pipeline with streaming_mode=async_audit should build without error"
    );
}
