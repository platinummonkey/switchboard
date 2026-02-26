pub mod builtin;
pub mod engine;
pub mod grpc_callout;
pub mod http_callout;
pub mod pipeline;

/// Generated protobuf / tonic code for the guardrail evaluator service.
pub mod proto {
    pub mod guardrails_v1 {
        tonic::include_proto!("guardrails.v1");
    }
}

pub use builtin::engine_from_config;
pub use engine::{
    AuditSeverity, GuardrailAction, GuardrailEngine, GuardrailInput, GuardrailVerdict,
};
pub use grpc_callout::{GrpcCalloutEngine, async_engine_from_config};
pub use http_callout::HttpCalloutEngine;
pub use pipeline::{FailMode, GuardrailPipeline, StreamingMode};
