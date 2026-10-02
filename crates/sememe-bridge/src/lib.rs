//! PyO3 bridge to a live PyTorch model.
//!
//! M2 ships `load` and `named_modules`. M3 adds `run_forward` (hook firing,
//! Arrow IPC). M5 adds `edit`. The bridge owns the FFI boundary; the
//! `sememe` core stays pure Rust.

use std::path::Path;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyModuleMethods};

use sememe::backend::Backend;
use sememe::{
    EditOp, Error as CoreError, ModelInput, ModulePath, ModuleTree, Result as CoreResult,
    TensorView,
};

/// Wrap a `transformers.PreTrainedModel` so the rest of the bridge can
/// call into it. The shim already returned a fully-loaded model; this
/// type just holds a `PyObject` handle.
#[pyclass]
pub struct PyModel {
    inner: PyObject,
}

#[pymethods]
impl PyModel {
    /// Depth-first list of every module's dotted name.
    pub fn named_modules<'py>(&self, py: Python<'py>) -> PyResult<Vec<String>> {
        let iter = self.inner.call_method0(py, "named_modules")?;
        let mut out = Vec::new();
        loop {
            let next = iter.call_method0(py, "__next__");
            match next {
                Ok(item) => {
                    let bound = item.bind(py);
                    let name: String = bound.get_item(0)?.extract()?;
                    if !name.is_empty() {
                        out.push(name);
                    }
                }
                Err(_) => break,
            }
        }
        Ok(out)
    }
}

/// Load a PyTorch model from a HuggingFace cache snapshot directory.
#[pyfunction]
pub fn load(py: Python<'_>, path: &str) -> PyResult<PyModel> {
    let p = Path::new(path);
    if !p.is_dir() {
        return Err(PyException::new_err(format!(
            "model path is not a directory: {path}"
        )));
    }
    let transformers = py.import("transformers")?;
    let auto_model = transformers.getattr("AutoModel")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", "bfloat16")?;
    let model = auto_model.call_method("from_pretrained", (path,), Some(&kwargs))?;
    Ok(PyModel {
        inner: model.unbind(),
    })
}

/// Return the Qwen snapshot path the integration tests use. Lives in
/// Rust so tests share one source of truth.
#[pyfunction]
pub fn demo_model_path() -> &'static str {
    "/Users/ericmey/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
}

#[pymodule]
fn sememe_bridge(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(demo_model_path, m)?)?;
    m.add_class::<PyModel>()?;
    Ok(())
}

// ---- Rust-side typed handle -----------------------------------------------

/// Type-safe Rust handle for a `PyModel`. Implements `Backend` so the
/// `Harness` can drive it without knowing about Python.
pub struct PyBackend {
    model: Py<PyModel>,
}

impl PyBackend {
    /// Load a model and wrap it.
    pub fn load(path: &str) -> CoreResult<Self> {
        Python::with_gil(|py| {
            let model = load(py, path).map_err(|e| CoreError::Backend(e.to_string()))?;
            let model = Py::new(py, model).map_err(|e| CoreError::Backend(e.to_string()))?;
            Ok(Self { model })
        })
    }
}

impl Backend for PyBackend {
    fn named_modules(&self) -> CoreResult<ModuleTree> {
        let names: Vec<String> = Python::with_gil(|py| {
            let m = self.model.borrow(py);
            m.named_modules(py)
                .map_err(|e| CoreError::Backend(e.to_string()))
        })?;
        let children: Vec<ModulePath> = names.into_iter().map(ModulePath::new).collect();
        let root = children
            .first()
            .map(|p| p.as_str().split('.').next().unwrap_or("").to_string())
            .unwrap_or_default();
        Ok(ModuleTree { root, children })
    }

    fn run_forward(&self, _input: &ModelInput) -> CoreResult<Vec<TensorView>> {
        Err(CoreError::Unsupported(
            "Backend::run_forward is not implemented in M2; lands in M3",
        ))
    }

    fn edit(&mut self, _path: &ModulePath, _op: &EditOp) -> CoreResult<()> {
        Err(CoreError::Unsupported(
            "Backend::edit is not implemented in M2; lands in M5",
        ))
    }
}
