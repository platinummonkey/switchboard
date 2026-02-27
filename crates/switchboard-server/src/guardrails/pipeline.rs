//! Guardrail pipeline orchestrator.
//!
//! [`GuardrailPipeline`] runs a list of [`GuardrailEngine`]s in order.
//! The first non-Pass verdict wins (fail-fast).  Each engine call is wrapped in
//! a per-engine timeout; on timeout or engine error the pipeline follows the
//! configured [`FailMode`].

use std::time::Duration;

use tracing::{debug, error, warn};

use crate::config::guardrails::GuardrailsConfig;
use crate::error::ServerError;
use crate::guardrails::builtin::engine_from_config;
use crate::guardrails::engine::{
    GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};
use crate::guardrails::grpc_callout::async_engine_from_config;

// ── FailMode ──────────────────────────────────────────────────────────────────

/// What to do when a guardrail engine errors or times out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailMode {
    /// Treat errors/timeouts as Pass and continue the pipeline.
    Open,
    /// Treat errors/timeouts as an immediate Block verdict.
    Closed,
}

impl FailMode {
    fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "closed" => FailMode::Closed,
            _ => FailMode::Open,
        }
    }
}

// ── StreamingMode ─────────────────────────────────────────────────────────────

/// How post-response guardrails handle streaming responses.
///
/// Not yet wired to the streaming handler — defined here for completeness and
/// future use by Phase 10b.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    /// Buffer the full response before evaluating.
    Buffered,
    /// Evaluate each chunk as it arrives.
    Chunked,
    /// Allow chunks through immediately; run evaluation asynchronously and
    /// emit an audit log if needed.
    AsyncAudit,
}

impl StreamingMode {
    fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "buffered" => StreamingMode::Buffered,
            "chunked" => StreamingMode::Chunked,
            _ => StreamingMode::AsyncAudit,
        }
    }
}

// ── GuardrailPipeline ─────────────────────────────────────────────────────────

/// Ordered pipeline of guardrail engines.
///
/// Pre-request engines run before the prompt is forwarded to the LLM.
/// Post-response engines run after the LLM response is received.
pub struct GuardrailPipeline {
    pre_request_engines: Vec<Box<dyn GuardrailEngine>>,
    post_response_engines: Vec<Box<dyn GuardrailEngine>>,
    fail_mode: FailMode,
    timeout: Duration,
    #[allow(dead_code)]
    streaming_mode: StreamingMode,
}

impl std::fmt::Debug for GuardrailPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailPipeline")
            .field("pre_request_engines_count", &self.pre_request_engines.len())
            .field(
                "post_response_engines_count",
                &self.post_response_engines.len(),
            )
            .field("fail_mode", &self.fail_mode)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Parse a human-friendly duration string (e.g. `"500ms"`, `"1s"`) into a
/// [`Duration`].  Falls back to 500 ms on any parse error.
fn parse_duration(s: &str) -> Duration {
    let s = s.trim();
    if let Some(ms_str) = s.strip_suffix("ms") {
        if let Ok(ms) = ms_str.trim().parse::<u64>() {
            return Duration::from_millis(ms);
        }
    }
    if let Some(s_str) = s.strip_suffix('s') {
        if let Ok(secs) = s_str.trim().parse::<u64>() {
            return Duration::from_secs(secs);
        }
    }
    Duration::from_millis(500) // sensible default
}

impl GuardrailPipeline {
    /// Build a pipeline from a [`GuardrailsConfig`].
    ///
    /// Engines in the config whose `phase` is `"pre_request"` go into
    /// `pre_request_engines`; those with `"post_response"` go into
    /// `post_response_engines`.
    ///
    /// This synchronous constructor only supports builtin and HTTP callout
    /// engine types.  For gRPC callout engines use [`Self::from_config_async`].
    pub fn from_config(cfg: &GuardrailsConfig) -> Result<Self, ServerError> {
        let fail_mode = FailMode::from_str(&cfg.fail_mode);
        let timeout = parse_duration(&cfg.timeout);
        let streaming_mode = StreamingMode::from_str(&cfg.streaming_mode);

        let mut pre_request_engines: Vec<Box<dyn GuardrailEngine>> = Vec::new();
        let mut post_response_engines: Vec<Box<dyn GuardrailEngine>> = Vec::new();

        for engine_cfg in &cfg.engines {
            let engine = engine_from_config(engine_cfg)?;
            match engine_cfg.phase.trim().to_ascii_lowercase().as_str() {
                "post_response" => post_response_engines.push(engine),
                _ => pre_request_engines.push(engine),
            }
        }

        Ok(Self {
            pre_request_engines,
            post_response_engines,
            fail_mode,
            timeout,
            streaming_mode,
        })
    }

    /// Async variant of [`Self::from_config`] that also supports the `"grpc"`
    /// engine type (which requires an async connection step).
    ///
    /// For each engine entry the method first tries the async factory
    /// (`async_engine_from_config`); if the entry is a builtin or HTTP type it
    /// falls back to the sync factory so that only one code path needs to be
    /// maintained.
    pub async fn from_config_async(cfg: &GuardrailsConfig) -> Result<Self, ServerError> {
        let fail_mode = FailMode::from_str(&cfg.fail_mode);
        let timeout = parse_duration(&cfg.timeout);
        let streaming_mode = StreamingMode::from_str(&cfg.streaming_mode);

        let mut pre_request_engines: Vec<Box<dyn GuardrailEngine>> = Vec::new();
        let mut post_response_engines: Vec<Box<dyn GuardrailEngine>> = Vec::new();

        for engine_cfg in &cfg.engines {
            // async_engine_from_config handles "grpc"; for all other types it
            // delegates to the sync builtin factory.
            let engine = async_engine_from_config(engine_cfg).await?;
            match engine_cfg.phase.trim().to_ascii_lowercase().as_str() {
                "post_response" => post_response_engines.push(engine),
                _ => pre_request_engines.push(engine),
            }
        }

        Ok(Self {
            pre_request_engines,
            post_response_engines,
            fail_mode,
            timeout,
            streaming_mode,
        })
    }

    /// Run all pre-request engines in order.
    ///
    /// Returns the first non-Pass verdict, or a Pass if all engines pass.
    pub async fn evaluate_request(&self, input: &GuardrailInput) -> GuardrailVerdict {
        self.run_pipeline(&self.pre_request_engines, input, "pre_request")
            .await
    }

    /// Run all post-response engines in order.
    pub async fn evaluate_response(&self, input: &GuardrailInput) -> GuardrailVerdict {
        self.run_pipeline(&self.post_response_engines, input, "post_response")
            .await
    }

    async fn run_pipeline(
        &self,
        engines: &[Box<dyn GuardrailEngine>],
        input: &GuardrailInput,
        phase: &str,
    ) -> GuardrailVerdict {
        for engine in engines {
            let verdict = self.run_engine(engine.as_ref(), input, phase).await;
            if !verdict.action.is_pass() {
                return verdict;
            }
        }
        GuardrailVerdict::pass("pipeline")
    }

    async fn run_engine(
        &self,
        engine: &dyn GuardrailEngine,
        input: &GuardrailInput,
        phase: &str,
    ) -> GuardrailVerdict {
        let engine_name = engine.name().to_owned();

        // We need to call the right method depending on the phase.
        let result = match phase {
            "post_response" => {
                tokio::time::timeout(self.timeout, engine.evaluate_response(input)).await
            }
            _ => tokio::time::timeout(self.timeout, engine.evaluate_request(input)).await,
        };

        match result {
            // Engine returned in time with Ok verdict
            Ok(Ok(verdict)) => {
                debug!(
                    engine = %engine_name,
                    action = %verdict.action,
                    "engine evaluated"
                );
                verdict
            }
            // Engine returned in time but with an error
            Ok(Err(err)) => {
                error!(engine = %engine_name, error = %err, "guardrail engine error");
                match self.fail_mode {
                    FailMode::Open => {
                        warn!(engine = %engine_name, "fail-open: treating error as pass");
                        GuardrailVerdict::pass(&engine_name)
                    }
                    FailMode::Closed => {
                        error!(engine = %engine_name, "fail-closed: blocking request due to engine error");
                        GuardrailVerdict {
                            action: GuardrailAction::Block {
                                message: format!(
                                    "Guardrail engine '{engine_name}' error: request blocked"
                                ),
                            },
                            engine: engine_name.clone(),
                            rule: "engine_error".into(),
                            reason: Some(format!("Engine '{engine_name}' returned an error")),
                            confidence: 1.0,
                            latency: Duration::ZERO,
                        }
                    }
                }
            }
            // Engine timed out
            Err(_elapsed) => {
                warn!(engine = %engine_name, timeout_ms = ?self.timeout.as_millis(), "guardrail engine timed out");
                match self.fail_mode {
                    FailMode::Open => {
                        warn!(engine = %engine_name, "fail-open: treating timeout as pass");
                        GuardrailVerdict::pass(&engine_name)
                    }
                    FailMode::Closed => {
                        error!(engine = %engine_name, "fail-closed: blocking request due to timeout");
                        GuardrailVerdict {
                            action: GuardrailAction::Block {
                                message: format!(
                                    "Guardrail engine '{engine_name}' timed out: request blocked"
                                ),
                            },
                            engine: engine_name.clone(),
                            rule: "engine_timeout".into(),
                            reason: Some(format!("Engine '{engine_name}' exceeded timeout")),
                            confidence: 1.0,
                            latency: self.timeout,
                        }
                    }
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;

    use super::*;
    use crate::config::guardrails::{EngineConfig, GuardrailsConfig, RegexRule};

    // ── Helper types ──────────────────────────────────────────────────────────

    fn make_input(content: &str) -> GuardrailInput {
        GuardrailInput {
            content: content.into(),
            messages: vec![],
            user: None,
            model: "claude-sonnet-4-20250514".into(),
            metadata: HashMap::new(),
        }
    }

    struct AlwaysPassEngine;
    #[async_trait]
    impl GuardrailEngine for AlwaysPassEngine {
        fn name(&self) -> &str {
            "always_pass"
        }
        async fn evaluate_request(
            &self,
            _: &GuardrailInput,
        ) -> Result<GuardrailVerdict, ServerError> {
            Ok(GuardrailVerdict::pass(self.name()))
        }
        async fn evaluate_response(
            &self,
            _: &GuardrailInput,
        ) -> Result<GuardrailVerdict, ServerError> {
            Ok(GuardrailVerdict::pass(self.name()))
        }
    }

    struct AlwaysBlockEngine;
    #[async_trait]
    impl GuardrailEngine for AlwaysBlockEngine {
        fn name(&self) -> &str {
            "always_block"
        }
        async fn evaluate_request(
            &self,
            _: &GuardrailInput,
        ) -> Result<GuardrailVerdict, ServerError> {
            Ok(GuardrailVerdict {
                action: GuardrailAction::Block {
                    message: "always blocked".into(),
                },
                engine: self.name().into(),
                rule: "always".into(),
                reason: None,
                confidence: 1.0,
                latency: Duration::ZERO,
            })
        }
        async fn evaluate_response(
            &self,
            _: &GuardrailInput,
        ) -> Result<GuardrailVerdict, ServerError> {
            Ok(GuardrailVerdict::pass(self.name()))
        }
    }

    struct AlwaysErrorEngine;
    #[async_trait]
    impl GuardrailEngine for AlwaysErrorEngine {
        fn name(&self) -> &str {
            "always_error"
        }
        async fn evaluate_request(
            &self,
            _: &GuardrailInput,
        ) -> Result<GuardrailVerdict, ServerError> {
            Err(ServerError::Config("intentional test error".into()))
        }
        async fn evaluate_response(
            &self,
            _: &GuardrailInput,
        ) -> Result<GuardrailVerdict, ServerError> {
            Err(ServerError::Config("intentional test error".into()))
        }
    }

    struct SleepyEngine {
        sleep: Duration,
    }
    #[async_trait]
    impl GuardrailEngine for SleepyEngine {
        fn name(&self) -> &str {
            "sleepy"
        }
        async fn evaluate_request(
            &self,
            _: &GuardrailInput,
        ) -> Result<GuardrailVerdict, ServerError> {
            tokio::time::sleep(self.sleep).await;
            Ok(GuardrailVerdict::pass(self.name()))
        }
        async fn evaluate_response(
            &self,
            _: &GuardrailInput,
        ) -> Result<GuardrailVerdict, ServerError> {
            tokio::time::sleep(self.sleep).await;
            Ok(GuardrailVerdict::pass(self.name()))
        }
    }

    // ── Helper: build pipeline directly ──────────────────────────────────────

    fn pipeline_with_pre(
        engines: Vec<Box<dyn GuardrailEngine>>,
        fail_mode: FailMode,
        timeout: Duration,
    ) -> GuardrailPipeline {
        GuardrailPipeline {
            pre_request_engines: engines,
            post_response_engines: vec![],
            fail_mode,
            timeout,
            streaming_mode: StreamingMode::AsyncAudit,
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_pipeline_all_pass_returns_pass() {
        let pipeline = pipeline_with_pre(
            vec![Box::new(AlwaysPassEngine), Box::new(AlwaysPassEngine)],
            FailMode::Open,
            Duration::from_secs(1),
        );
        let verdict = pipeline.evaluate_request(&make_input("hello")).await;
        assert!(verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_pipeline_first_block_wins() {
        let pipeline = pipeline_with_pre(
            vec![Box::new(AlwaysBlockEngine), Box::new(AlwaysPassEngine)],
            FailMode::Open,
            Duration::from_secs(1),
        );
        let verdict = pipeline.evaluate_request(&make_input("hello")).await;
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.engine, "always_block");
    }

    #[tokio::test]
    async fn test_pipeline_fail_open_on_engine_error() {
        let pipeline = pipeline_with_pre(
            vec![Box::new(AlwaysErrorEngine)],
            FailMode::Open,
            Duration::from_secs(1),
        );
        let verdict = pipeline.evaluate_request(&make_input("hello")).await;
        // Fail-open: error treated as pass
        assert!(verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_pipeline_fail_closed_on_engine_error() {
        let pipeline = pipeline_with_pre(
            vec![Box::new(AlwaysErrorEngine)],
            FailMode::Closed,
            Duration::from_secs(1),
        );
        let verdict = pipeline.evaluate_request(&make_input("hello")).await;
        assert!(verdict.action.is_blocking());
        assert_eq!(verdict.rule, "engine_error");
    }

    #[tokio::test]
    async fn test_pipeline_timeout_fail_open() {
        let pipeline = pipeline_with_pre(
            vec![Box::new(SleepyEngine {
                sleep: Duration::from_millis(200),
            })],
            FailMode::Open,
            Duration::from_millis(10), // very short timeout
        );
        let verdict = pipeline.evaluate_request(&make_input("hello")).await;
        assert!(verdict.action.is_pass(), "fail-open timeout should pass");
    }

    #[tokio::test]
    async fn test_pipeline_timeout_fail_closed() {
        let pipeline = pipeline_with_pre(
            vec![Box::new(SleepyEngine {
                sleep: Duration::from_millis(200),
            })],
            FailMode::Closed,
            Duration::from_millis(10),
        );
        let verdict = pipeline.evaluate_request(&make_input("hello")).await;
        assert!(
            verdict.action.is_blocking(),
            "fail-closed timeout should block"
        );
        assert_eq!(verdict.rule, "engine_timeout");
    }

    #[tokio::test]
    async fn test_pipeline_pre_vs_post_separation() {
        // BlockEngine is only in post, pre should pass
        let pipeline = GuardrailPipeline {
            pre_request_engines: vec![Box::new(AlwaysPassEngine)],
            post_response_engines: vec![Box::new(AlwaysBlockEngine)],
            fail_mode: FailMode::Open,
            timeout: Duration::from_secs(1),
            streaming_mode: StreamingMode::AsyncAudit,
        };

        let pre_verdict = pipeline.evaluate_request(&make_input("hello")).await;
        assert!(pre_verdict.action.is_pass(), "pre should pass");

        // Post-response evaluate_response on AlwaysBlockEngine passes (see impl)
        // Let's instead test that post engines don't run during evaluate_request
        let post_verdict = pipeline.evaluate_response(&make_input("hello")).await;
        // AlwaysBlockEngine.evaluate_response returns pass (see impl above)
        assert!(post_verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_pipeline_from_config_regex_engine() {
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

        let pipeline = GuardrailPipeline::from_config(&cfg).unwrap();
        assert_eq!(pipeline.pre_request_engines.len(), 1);
        assert_eq!(pipeline.post_response_engines.len(), 0);

        let bad_input = make_input("My SSN is 123-45-6789");
        let verdict = pipeline.evaluate_request(&bad_input).await;
        assert!(verdict.action.is_blocking());

        let clean_input = make_input("What is the capital of France?");
        let clean_verdict = pipeline.evaluate_request(&clean_input).await;
        assert!(clean_verdict.action.is_pass());
    }

    #[tokio::test]
    async fn test_pipeline_from_config_secret_detection() {
        let cfg = GuardrailsConfig {
            enabled: true,
            fail_mode: "closed".into(),
            timeout: "500ms".into(),
            streaming_mode: "buffered".into(),
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

        let pipeline = GuardrailPipeline::from_config(&cfg).unwrap();
        let secret_input =
            make_input("Here is my key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456789");
        let verdict = pipeline.evaluate_request(&secret_input).await;
        assert!(verdict.action.is_blocking());
    }

    #[test]
    fn test_parse_duration_ms() {
        assert_eq!(parse_duration("500ms"), Duration::from_millis(500));
        assert_eq!(parse_duration("100ms"), Duration::from_millis(100));
    }

    #[test]
    fn test_parse_duration_s() {
        assert_eq!(parse_duration("1s"), Duration::from_secs(1));
        assert_eq!(parse_duration("30s"), Duration::from_secs(30));
    }

    #[test]
    fn test_parse_duration_fallback() {
        assert_eq!(parse_duration("invalid"), Duration::from_millis(500));
        assert_eq!(parse_duration(""), Duration::from_millis(500));
    }

    #[test]
    fn test_fail_mode_from_str() {
        assert_eq!(FailMode::from_str("open"), FailMode::Open);
        assert_eq!(FailMode::from_str("closed"), FailMode::Closed);
        assert_eq!(FailMode::from_str("OPEN"), FailMode::Open);
        assert_eq!(FailMode::from_str("unknown"), FailMode::Open); // default
    }

    #[test]
    fn test_streaming_mode_from_str() {
        assert_eq!(StreamingMode::from_str("buffered"), StreamingMode::Buffered);
        assert_eq!(StreamingMode::from_str("chunked"), StreamingMode::Chunked);
        assert_eq!(
            StreamingMode::from_str("async_audit"),
            StreamingMode::AsyncAudit
        );
        assert_eq!(
            StreamingMode::from_str("unknown"),
            StreamingMode::AsyncAudit
        );
    }
}
