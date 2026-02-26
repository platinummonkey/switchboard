use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Wraps both `tonic::transport::Error` and `tonic::Status` (both are
    /// large types; boxing keeps the enum variant size small so that
    /// `clippy::result_large_err` does not fire).
    #[error("gRPC error: {0}")]
    Grpc(#[from] Box<GrpcError>),
}

/// Inner detail for gRPC errors, kept behind a `Box` in `ServerError`.
#[derive(Debug, Error)]
pub enum GrpcError {
    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("status error: {0}")]
    Status(#[from] tonic::Status),
}

// ── Convenience From impls ────────────────────────────────────────────────────

impl From<tonic::transport::Error> for ServerError {
    fn from(e: tonic::transport::Error) -> Self {
        ServerError::Grpc(Box::new(GrpcError::Transport(e)))
    }
}

impl From<tonic::Status> for ServerError {
    fn from(e: tonic::Status) -> Self {
        ServerError::Grpc(Box::new(GrpcError::Status(e)))
    }
}
