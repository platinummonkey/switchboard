use thiserror::Error;

#[derive(Debug, Error)]
pub enum LocalError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("authentication error: {0}")]
    Auth(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
