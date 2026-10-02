use crate::edit::EditOp;
use crate::error::Result;
use crate::input::ModelInput;
use crate::path::ModulePath;
use crate::topology::ModuleTree;
use crate::view::TensorView;

/// The seam between `sememe`'s core and any model backend. M1 declares
/// the full future surface so M2/M3/M5 add implementations, not new
/// methods. The stub implementation in `backend::stub` returns
/// `Error::Unsupported` for methods it doesn't yet model.
pub trait Backend {
    /// One-shot: list the model's named modules. Called once at attach
    /// time and again whenever the topology is believed to have changed.
    fn named_modules(&self) -> Result<ModuleTree>;

    /// Run a forward pass and stream every hook's `TensorView`. The M3
    /// bridge returns one view per hooked module.
    fn run_forward(&self, input: &ModelInput) -> Result<Vec<TensorView>>;

    /// Apply an `EditOp` at a named module path. M5's bridge performs
    /// the actual mutation in PyTorch.
    fn edit(&mut self, path: &ModulePath, op: &EditOp) -> Result<()>;
}

#[cfg(any(test, feature = "test-util"))]
pub mod stub;
