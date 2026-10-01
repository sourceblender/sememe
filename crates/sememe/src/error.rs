use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("backend error: {0}")]
    Backend(String),

    #[error("invalid module path: {0}")]
    InvalidPath(String),

    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
