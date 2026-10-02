use serde::{Deserialize, Serialize};

/// Dtype of an observed tensor. Deliberately small — the bridge maps
/// anything exotic to one of these in M3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorDType {
    Float32,
    Float16,
    Bfloat16,
    Int64,
    Bool,
}

impl TensorDType {
    pub fn name(self) -> &'static str {
        match self {
            TensorDType::Float32 => "f32",
            TensorDType::Float16 => "f16",
            TensorDType::Bfloat16 => "bf16",
            TensorDType::Int64 => "i64",
            TensorDType::Bool => "bool",
        }
    }
}
