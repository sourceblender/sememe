//! sememe — debug harness for text-encoder models.
//!
//! M1 ships the core types in pure Rust: a `Backend` seam, an in-memory
//! `Telemetry` hub, a generic `Harness` that wires them together, and the
//! serializable `RecordedEvent` schema used by the M4 NDJSON writer and
//! the M6 replay backend. No PyTorch bridge or TUI yet.

pub mod backend;
pub mod dtype;
pub mod edit;
pub mod error;
pub mod harness;
pub mod hub;
pub mod input;
pub mod path;
pub mod recorded;
pub mod telemetry;
pub mod topology;
pub mod view;

pub use backend::Backend;
pub use dtype::TensorDType;
pub use edit::EditOp;
pub use error::{Error, Result};
pub use harness::Harness;
pub use hub::{HubEvent, SubscriptionId};
pub use input::ModelInput;
pub use path::ModulePath;
pub use recorded::RecordedEvent;
pub use telemetry::{ModuleNode, Telemetry};
pub use topology::ModuleTree;
pub use view::TensorView;

/// Crate version, sourced from `Cargo.toml` at build time.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
