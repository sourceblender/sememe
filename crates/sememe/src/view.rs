use serde::{Deserialize, Serialize};

use crate::dtype::TensorDType;
use crate::path::ModulePath;

/// Opaque stats record for an observed tensor. Deliberately not the raw
/// tensor — the raw tensor lives on the Python side (M2+). Observing and
/// patching stats is enough to reason about behaviour and cheap enough to
/// record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorView {
    pub path: ModulePath,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub samples: Vec<f64>,
    /// Index of the forward pass that produced this view, recorded by
    /// `Telemetry`. `0` for views constructed outside the hub (e.g. in
    /// tests). The hub overwrites this on `record_observation`.
    pub forward_index: u64,
}

impl TensorView {
    /// Compute stats from a flat slice of f32 values, treating `shape` as
    /// the tensor's shape. Used by the in-memory stub backend; the M3
    /// bridge will construct `TensorView` from PyTorch tensors via the
    /// same math.
    pub fn from_slice(
        path: ModulePath,
        shape: Vec<usize>,
        dtype: TensorDType,
        values: &[f32],
    ) -> Self {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0_f64;
        for &v in values {
            let v = v as f64;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
        }
        let mean = if values.is_empty() {
            0.0
        } else {
            sum / values.len() as f64
        };

        let samples: Vec<f64> = values.iter().take(8).map(|v| *v as f64).collect();

        Self {
            path,
            shape,
            dtype,
            min,
            max,
            mean,
            samples,
            forward_index: 0,
        }
    }
}
