use serde::{Deserialize, Serialize};

use crate::view::TensorView;

/// A surgical edit applied to a module's output during a forward pass.
/// Variants are deliberately small — each one maps to a concrete operation
/// the M5 bridge performs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EditOp {
    Zero,
    Scale(f64),
    Add(f64),
    Patch(TensorView),
}
