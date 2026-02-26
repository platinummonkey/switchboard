pub mod builtin;
pub mod engine;
pub mod http_callout;
pub mod pipeline;

pub use builtin::engine_from_config;
pub use engine::{
    AuditSeverity, GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};
pub use http_callout::HttpCalloutEngine;
pub use pipeline::{FailMode, GuardrailPipeline, StreamingMode};

use crate::config::guardrails::EngineConfig;
use crate::error::ServerError;

/// Build an engine that may require async setup (grpc, http).
/// Falls back to `engine_from_config` for builtin types.
///
/// This function exists for API symmetry with a potential gRPC async factory;
/// HTTP client construction is synchronous so no real async work is done here.
pub async fn async_engine_from_config(
    cfg: &EngineConfig,
) -> Result<Box<dyn GuardrailEngine>, ServerError> {
    engine_from_config(cfg)
}
