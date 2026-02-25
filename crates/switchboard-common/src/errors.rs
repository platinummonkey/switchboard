use thiserror::Error;

#[derive(Debug, Error)]
pub enum SwitchboardError {
    #[error("authentication error: {0}")]
    Auth(String),

    #[error("upstream provider error: {0}")]
    Upstream(String),

    #[error("key pool exhausted")]
    KeyPoolExhausted,

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("guardrail blocked request: {reason}")]
    GuardrailBlocked { reason: String },

    #[error("rate limit exceeded")]
    RateLimitExceeded,

    #[error("configuration error: {0}")]
    Config(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("internal error: {0}")]
    Internal(String),
}
