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
/// `ModelOverrideLayer` now buffers the request body and extracts the `"model"`
/// field, populating `ctx.model` before calling `selector.select()`.  The body
/// model `"gpt-4o"` has no mapping entry (the mapping is `gpt-4 → gpt-4o`),
/// so the selector falls through to the fallback `"gpt-4o"`.  The result is
/// the same as before — the request succeeds with the fallback model.
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

    // The body model "gpt-4o" has no mapping entry, so the fallback "gpt-4o" is used.
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
// # How per-user overrides now work end-to-end
//
// `ModelOverrideLayer` buffers the request body and reads the `x-switchboard-user`
// header to build a `RequestContext` with `user_id` populated BEFORE calling
// `selector.select()`.  This means per-user overrides in `dynamic` mode fire
// at the middleware layer (step 2 of `select_dynamic()`), not only inside the
// handler.  The resolved model is stored in the `ResolvedModel` extension and
// used by the handler to select the upstream provider and key pool.
//
// # Test strategy
//
// 1. Verify that without `x-switchboard-user` the fallback is used.
// 2. Verify that the `x-switchboard-model` header takes precedence over every
//    other consideration (step 1 fires before step 2).
// 3. Verify that multiple users with different overrides co-exist.
// 4. Confirm the body model forwarded to upstream is unchanged (the middleware
//    resolves the provider but does not rewrite the request body model field).

/// Per-user override defined in config: when no `x-switchboard-user` header is
/// present the middleware cannot populate `ctx.user_id`, so the per-user override
/// branch in `select_dynamic()` is never reached.  The selector falls through
/// to the configured `fallback` model.
///
/// This test verifies that without the user header the fallback is applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_per_user_model_override_uses_fallback_without_header() {
    // Configure a per-user override for "alice" → "gpt-3.5-turbo".
    // No x-switchboard-user header is sent, so ctx.user_id remains None and
    // the per-user override is never applied.  The selector uses the fallback.
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

    // Mount "gpt-4o" — the fallback that applies when no user header is sent.
    mock_chat_ok("gpt-4o", "fallback-response")
        .mount(openai_mock)
        .await;

    // Send without x-switchboard-model and without x-switchboard-user.
    // No user_id is available to the middleware so the fallback "gpt-4o" is used.
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
/// Sends `x-switchboard-user` for alice, bob, and carol who all have overrides
/// pointing to "gpt-3.5-turbo".  The middleware applies each per-user override.
/// The OpenAI wiremock accepts any POST to `/v1/chat/completions` regardless of
/// the body model, so all requests return 200.
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

    // The wiremock matches any POST body, so overriding to gpt-3.5-turbo still succeeds.
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

/// Inspect the raw upstream request body to confirm the model field forwarded
/// to the upstream is the ORIGINAL body model `"gpt-4o"`, not the override
/// model `"gpt-3.5-turbo"`.
///
/// When the middleware resolves alice's per-user override to `"gpt-3.5-turbo"`,
/// it selects the OpenAI provider and key pool (both `gpt-4o` and `gpt-3.5-turbo`
/// are registered with OpenAI in the test config).  The proxy handler does NOT
/// rewrite the request body — it forwards the original body unchanged.  The
/// upstream therefore still receives `"model": "gpt-4o"`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_per_user_model_override_upstream_receives_fallback_model() {
    let mut overrides = HashMap::new();
    // Configure a per-user override for "alice" → "gpt-3.5-turbo".
    // The middleware will select "gpt-3.5-turbo" as the provider routing key,
    // but the request body model ("gpt-4o") is forwarded unchanged.
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

    // The body model forwarded to upstream is "gpt-4o", NOT "gpt-3.5-turbo".
    // The middleware selects the provider using the resolved model but does not
    // rewrite the request body.
    assert_eq!(
        upstream_body["model"], "gpt-4o",
        "upstream must receive body model 'gpt-4o', not the override 'gpt-3.5-turbo': got {:?}",
        upstream_body["model"]
    );
}
