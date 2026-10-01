use std::collections::BTreeMap;

use crate::backend::Backend;
use crate::dtype::TensorDType;
use crate::edit::EditOp;
use crate::error::{Error, Result};
use crate::input::ModelInput;
use crate::path::ModulePath;
use crate::topology::ModuleTree;
use crate::view::TensorView;

/// In-memory test backend. Holds a hand-built `ModuleTree` and a flat
/// `f32` vector per leaf path. `run_forward` reduces each vector into a
/// `TensorView`. `edit` mutates the stored vector in place. Useful in
/// tests and as the reference implementation of the seam.
pub struct StubBackend {
    tree: ModuleTree,
    tensors: BTreeMap<ModulePath, Vec<f32>>,
}

impl StubBackend {
    /// Build a stub from a tree and a map of leaf path → data.
    pub fn new(tree: ModuleTree, tensors: BTreeMap<ModulePath, Vec<f32>>) -> Self {
        Self { tree, tensors }
    }
}

impl Backend for StubBackend {
    fn named_modules(&self) -> Result<ModuleTree> {
        Ok(self.tree.clone())
    }

    fn run_forward(&self, _input: &ModelInput) -> Result<Vec<TensorView>> {
        let mut out = Vec::new();
        for (path, values) in &self.tensors {
            let view = TensorView::from_slice(
                path.clone(),
                vec![values.len()],
                TensorDType::Float32,
                values,
            );
            out.push(view);
        }
        Ok(out)
    }

    fn edit(&mut self, path: &ModulePath, op: &EditOp) -> Result<()> {
        let values = self
            .tensors
            .get_mut(path)
            .ok_or_else(|| Error::InvalidPath(path.to_string()))?;
        match op {
            EditOp::Zero => values.fill(0.0),
            EditOp::Scale(k) => {
                for v in values.iter_mut() {
                    *v *= *k as f32;
                }
            }
            EditOp::Add(k) => {
                for v in values.iter_mut() {
                    *v += *k as f32;
                }
            }
            EditOp::Patch(view) => {
                for (slot, sample) in values.iter_mut().zip(view.samples.iter()) {
                    *slot = *sample as f32;
                }
            }
        }
        Ok(())
    }
}
