//! sememe — debug harness for text-encoder models.

/// Crate version, sourced from `Cargo.toml` at build time.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Placeholder harness handle. Real API lands in M1.
pub struct Harness;
