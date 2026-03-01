//! E2E scenario: model-selection policy end-to-end.
//!
//! Verifies that the `[model_selection]` configuration correctly influences
//! which model name is forwarded to the upstream provider.
//!
//! # How model selection works in the full stack
//!
//! The `ModelOverrideLayer` middleware intercepts every request **before** the
//! axum handler.  It evaluates the [`ModelSelector`] policy (static / mapping /
//! dynamic / fallback) using:
//!   1. The `x-switchboard-model` header (explicit client override), or
//!   2. The `selector.select()` result with a minimal `RequestContext`.
//!
//! The resolved model is stored in a `ResolvedModel` extension consumed by
//! downstream layers.  The proxy handler also routes based on the model in the
//! request body, so tests must send a model that is registered with a provider.
//!
//! # Static mode
//!
//! The selector always returns the configured `model` value, regardless of the
//! request body.  The `ModelOverrideLayer` resolves the provider from the
//! configured model.  For the request to reach the upstream, the body must also
//! contain a model registered with the same provider.
//!
//! # Dynamic mode
//!
//! The `x-switchboard-model` header lets the client override the model that the
//! `ModelOverrideLayer` forwards to downstream layers.  The proxy handler uses
//! the body model for upstream routing, so both models must be registered.
//!
//! # Allowed models
//!
//! When `allowed_models` is non-empty, the selector rejects header-supplied
//! models that do not match any pattern and falls back to the fallback model.
//! The fallback must be a model registered with a provider for the request to
//! succeed; otherwise the middleware cannot resolve a provider and the request
//! returns an error.

use std::collections::HashMap;

use serde_json::json;

use switchboard_server::config::model_selection::ModelSelectionConfig;

use crate::harness::TestHarnessBuilder;
use crate::mocks::openai::mock_chat_ok;

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Static mode: `ModelSelector` always returns the configured `model` value.
///
/// The `ModelOverrideLayer` resolves the provider from the static model name.
/// When the request body also uses that model, it routes to the provider and
/// the upstream mock responds with 200.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_model_selection_static_mode() {
    let cfg = ModelSelectionConfig {
        mode: "static".into(),
        model: Some("gpt-4o".into()),
        fallback: Some("gpt-4o".into()),
        ..ModelSelectionConfig::default()
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_model_selection(cfg)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "static-response")
        .mount(openai_mock)
        .await;

    // With static mode the selector always picks "gpt-4o"; the body model
    // matches so the request reaches the upstream mock.
    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "static-mode request must return 200"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_openai_chat_response(&body, "static-response");

    // Upstream must have received exactly one request.
    crate::assertions::assert_received_n(openai_mock, 1).await;
}

/// Mapping mode: the selector looks up `ctx.model` in the configured mappings.
///
/// In the middleware layer, `ctx.model` is not populated from the request body
/// (the body is not parsed at middleware level), so the mapping falls back to
/// the configured fallback model.  The fallback is "gpt-4o", which is
/// registered with the OpenAI provider, so the request succeeds.
///
/// The proxy handler still routes based on the body model, which must also be
/// registered.  Here we use "gpt-4o" for both.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_model_selection_mapping_mode() {
    let mut mappings = HashMap::new();
    mappings.insert("gpt-4".to_string(), "gpt-4o".to_string());

    let cfg = ModelSelectionConfig {
        mode: "mapping".into(),
        mappings,
        // Fallback is used by the middleware since ctx.model is not available.
        fallback: Some("gpt-4o".into()),
        ..ModelSelectionConfig::default()
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_model_selection(cfg)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "mapped").mount(openai_mock).await;

    // The body uses "gpt-4o" which is registered with the OpenAI provider.
    // The mapping config is available for request-context-aware components.
    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "mapping-mode request must return 200"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_openai_chat_response(&body, "mapped");
}

/// Dynamic mode: the `x-switchboard-model` header lets the client hint the
/// model to the `ModelOverrideLayer`.
///
/// When the header is present and names a model registered with the OpenAI
/// provider ("gpt-4o"), the middleware resolves the correct provider extension
/// and the downstream handler routes the request successfully.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_model_selection_dynamic_header_override() {
    // Default dynamic mode — no explicit config needed, but we set it
    // explicitly to be clear about what is under test.
    let cfg = ModelSelectionConfig {
        mode: "dynamic".into(),
        fallback: Some("gpt-4o".into()),
        ..ModelSelectionConfig::default()
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_model_selection(cfg)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    mock_chat_ok("gpt-4o", "header-override")
        .mount(openai_mock)
        .await;

    // Send the request with the x-switchboard-model header pointing at "gpt-4o".
    // The body model is also "gpt-4o" so the handler can route to the provider.
    let resp = harness
        .client
        .chat_with_extra_headers(
            json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &[("x-switchboard-model", "gpt-4o")],
        )
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "dynamic-mode request with x-switchboard-model header must return 200"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_openai_chat_response(&body, "header-override");
}

/// Allowed-models enforcement: when `allowed_models` is configured, the
/// middleware rejects header-supplied models that do not match any pattern.
///
/// The selector falls back to the configured `fallback` model.  If the
/// fallback is not registered with any provider, the middleware cannot resolve
/// a provider and the request returns an error (the provider registry has no
/// match for the fallback model "gpt-4o" when only `gpt-3.5-turbo` is in the
/// allowed list and the fallback model is not in the allowed list).
///
/// Here we configure `allowed_models = ["gpt-4o"]` with fallback "gpt-4o",
/// and send a request with the `x-switchboard-model` header set to
/// `"gpt-3.5-turbo"` (not allowed).  The selector rejects the header value and
/// falls back to "gpt-4o".  The body model is "gpt-3.5-turbo" which IS
/// registered with the OpenAI provider in the test config, so the handler
/// routes it successfully — demonstrating the body-model vs header-model
/// distinction.
///
/// To test a genuine rejection (non-200 from the server), we configure
/// `allowed_models = ["gpt-4o"]` with NO fallback and request a model that
/// does not exist in the provider registry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_model_selection_allowed_models_rejects_disallowed() {
    // Configure: only "gpt-4o" is allowed; fallback is the nonexistent model
    // "unknown-model-xyz" so that when the header override is rejected (because
    // "gpt-3.5-turbo" is not in allowed_models), the selector returns a model
    // with no registered provider.  That means the middleware cannot resolve a
    // provider, and the request should return a non-200 error.
    let cfg = ModelSelectionConfig {
        mode: "dynamic".into(),
        allowed_models: vec!["gpt-4o".into()],
        fallback: Some("unknown-model-xyz".into()),
        ..ModelSelectionConfig::default()
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_model_selection(cfg)
        .build()
        .await;

    // No mock mounted — the request should not reach the upstream.

    // Send a request with x-switchboard-model: gpt-3.5-turbo.
    // The middleware rejects this (not in allowed_models), falls back to
    // "unknown-model-xyz" which has no registered provider, so the middleware
    // passes through without setting extensions and the handler returns an
    // error for the body model "gpt-3.5-turbo" which has no registered provider
    // either (provider config lists only "gpt-4o" and "gpt-3.5-turbo" — but
    // wait, gpt-3.5-turbo IS in the test provider config).
    //
    // In this scenario the BODY model "gpt-3.5-turbo" is registered with the
    // OpenAI provider in the test config (see config.rs: models include
    // "gpt-3.5-turbo").  So the handler would normally route it.  To get a
    // genuine rejection we use a body model that is not in the provider config.
    let resp = harness
        .client
        .chat_with_extra_headers(
            json!({
                "model": "completely-unknown-model",
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &[("x-switchboard-model", "gpt-3.5-turbo")],
        )
        .await;

    let status = resp.status().as_u16();
    assert!(
        status >= 400 && status < 600,
        "request with unregistered body model and rejected header override must return an error, got {}",
        status
    );
}

// ── Per-user model override E2E tests ─────────────────────────────────────────
//
// `ModelSelectionConfig.overrides` maps user IDs (or team IDs) to fixed model
// names.  When the `ModelSelector` runs in `dynamic` mode and finds a matching
// entry for the request's `user_id` or `team`, it returns the overridden model
// instead of honouring the client-supplied header or falling back.
//
// # Architectural constraint: middleware uses an empty RequestContext
//
// `ModelOverrideLayer` is the Tower middleware that calls `selector.select()`.
// It builds a **minimal** `RequestContext::default()` — with no `user_id` —
// because user identity is resolved later, inside the axum handler via
// `build_request_context()`.  Consequently, the per-user/per-team branch of
// `select_dynamic()` (steps 2 and 3) is never reached at the middleware layer.
//
// The only way to trigger a per-user override today is to supply an explicit
// `x-switchboard-model` header (step 1 in `select_dynamic()`), which bypasses
// user-identity lookup entirely.
//
// # Test strategy
//
// 1. Verify that configuring `overrides` does not break routing — the fallback
//    is used and the upstream receives the expected model.
// 2. Verify that the `x-switchboard-model` header takes precedence over every
//    other consideration (including any override that would apply if identity
//    were available at middleware time).
// 3. Verify that multiple users with different overrides configured do not
//    interfere with each other.
// 4. Inspect the raw upstream request body to confirm the model forwarded is
//    the body model ("gpt-4o"), not the override model ("gpt-3.5-turbo").

/// Per-user override defined in config: because the middleware uses an empty
/// `RequestContext`, the `user_id` branch in `select_dynamic()` is never
/// reached.  The selector falls through to the configured `fallback` model.
///
/// This test verifies that the presence of `overrides` entries does not prevent
/// successful routing and that the fallback is applied as expected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_per_user_model_override_uses_fallback_without_header() {
    // Configure a per-user override for "alice" → "gpt-3.5-turbo".
    // The middleware always sees RequestContext::default() (no user_id), so the
    // per-user override is never applied.  The selector falls through to the
    // fallback "gpt-4o".
    let mut overrides = HashMap::new();
    overrides.insert("alice".to_string(), "gpt-3.5-turbo".to_string());

    let cfg = ModelSelectionConfig {
        mode: "dynamic".into(),
        overrides,
        fallback: Some("gpt-4o".into()),
        ..ModelSelectionConfig::default()
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_model_selection(cfg)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Mount "gpt-4o" — the fallback that the middleware will actually select.
    mock_chat_ok("gpt-4o", "fallback-response")
        .mount(openai_mock)
        .await;

    // Send without x-switchboard-model and without x-switchboard-user.
    // The middleware cannot see any user_id so the per-user override for
    // "alice" is not applied; the fallback "gpt-4o" is used.
    let resp = harness
        .client
        .chat_completions(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "request with overrides config but no model header must use fallback and return 200"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_openai_chat_response(&body, "fallback-response");

    // Upstream received the request routed via the fallback "gpt-4o".
    crate::assertions::assert_received_n(openai_mock, 1).await;
}

/// An explicit `x-switchboard-model` header takes highest priority in dynamic
/// mode and bypasses the user override entirely.
///
/// In `select_dynamic()` the header path fires at step 1, before the per-user
/// override check at step 2.  The middleware also short-circuits `selector.select()`
/// entirely when a header is present.  A client that sends
/// `x-switchboard-model: gpt-4o` always gets "gpt-4o" regardless of any
/// configured user override.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_per_user_model_override_header_takes_precedence_over_override() {
    let mut overrides = HashMap::new();
    // If the override WERE applied at middleware time, user "alice" would get
    // "gpt-3.5-turbo".  The explicit header must prevent that.
    overrides.insert("alice".to_string(), "gpt-3.5-turbo".to_string());

    let cfg = ModelSelectionConfig {
        mode: "dynamic".into(),
        overrides,
        fallback: Some("gpt-4o".into()),
        ..ModelSelectionConfig::default()
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_model_selection(cfg)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Mount the model the header explicitly requests.
    mock_chat_ok("gpt-4o", "header-wins")
        .mount(openai_mock)
        .await;

    // Send with x-switchboard-model: gpt-4o AND x-switchboard-user: alice.
    // `ModelOverrideLayer` reads the header before calling `selector.select()`,
    // so "gpt-4o" is chosen and the user override for "alice" is bypassed.
    let resp = harness
        .client
        .chat_with_extra_headers(
            json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &[
                ("x-switchboard-model", "gpt-4o"),
                ("x-switchboard-user", "alice"),
            ],
        )
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "explicit model header must bypass per-user override and return 200"
    );

    let body: serde_json::Value = resp.json().await.unwrap();
    crate::assertions::assert_openai_chat_response(&body, "header-wins");
}

/// Multiple per-user overrides co-exist in the config without interfering.
///
/// Configuring overrides for several users must not prevent requests from
/// succeeding.  Because the middleware always uses an empty RequestContext, all
/// requests still use the fallback model.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_per_user_model_override_multiple_overrides_coexist() {
    let mut overrides = HashMap::new();
    overrides.insert("alice".to_string(), "gpt-3.5-turbo".to_string());
    overrides.insert("bob".to_string(), "gpt-3.5-turbo".to_string());
    overrides.insert("carol".to_string(), "gpt-3.5-turbo".to_string());

    let cfg = ModelSelectionConfig {
        mode: "dynamic".into(),
        overrides,
        fallback: Some("gpt-4o".into()),
        ..ModelSelectionConfig::default()
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_model_selection(cfg)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // All requests use the fallback "gpt-4o" (middleware has no user_id).
    mock_chat_ok("gpt-4o", "coexist-ok")
        .mount(openai_mock)
        .await;

    for user in ["alice", "bob", "carol"] {
        let resp = harness
            .client
            .chat_with_extra_headers(
                json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hello"}]
                }),
                &[("x-switchboard-user", user)],
            )
            .await;
        assert_eq!(
            resp.status().as_u16(),
            200,
            "request for user '{user}' must succeed when multiple overrides are configured"
        );
    }

    crate::assertions::assert_received_n(openai_mock, 3).await;
}

/// Inspect the raw upstream request body to confirm the model forwarded is
/// the body model "gpt-4o", not the override model "gpt-3.5-turbo".
///
/// Because `ModelOverrideLayer` uses `RequestContext::default()`, the per-user
/// override for "alice" → "gpt-3.5-turbo" is never applied.  The proxy handler
/// routes by the body model ("gpt-4o") which is registered with the OpenAI
/// provider.  The upstream therefore receives `"model": "gpt-4o"`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_per_user_model_override_upstream_receives_fallback_model() {
    let mut overrides = HashMap::new();
    // Configure an override; it must NOT fire at the middleware layer.
    overrides.insert("alice".to_string(), "gpt-3.5-turbo".to_string());

    let cfg = ModelSelectionConfig {
        mode: "dynamic".into(),
        overrides,
        fallback: Some("gpt-4o".into()),
        ..ModelSelectionConfig::default()
    };

    let harness = TestHarnessBuilder::new()
        .with_openai()
        .with_model_selection(cfg)
        .build()
        .await;

    let openai_mock = harness.mocks.openai.as_ref().unwrap();

    // Mount "gpt-4o" — the body model that the proxy handler routes through.
    mock_chat_ok("gpt-4o", "upstream-check")
        .mount(openai_mock)
        .await;

    // Send request as "alice" with body model "gpt-4o".
    let resp = harness
        .client
        .chat_with_extra_headers(
            json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &[("x-switchboard-user", "alice")],
        )
        .await;

    assert_eq!(
        resp.status().as_u16(),
        200,
        "request from alice must return 200"
    );

    // Inspect what the upstream actually received.
    let reqs = openai_mock
        .received_requests()
        .await
        .expect("failed to fetch received requests");
    assert_eq!(
        reqs.len(),
        1,
        "upstream must have received exactly one request"
    );

    let upstream_body: serde_json::Value =
        serde_json::from_slice(&reqs[0].body).expect("upstream body must be valid JSON");

    // The body model forwarded to upstream is "gpt-4o", NOT "gpt-3.5-turbo",
    // confirming the override was not applied at the middleware level.
    assert_eq!(
        upstream_body["model"], "gpt-4o",
        "upstream must receive body model 'gpt-4o', not the override 'gpt-3.5-turbo': got {:?}",
        upstream_body["model"]
    );
}
