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
    ///
    /// Iterates with Python's own iterator protocol, so only `StopIteration`
    /// ends the walk. Any other error from the model is returned, never
    /// mistaken for the end of the tree.
    pub fn named_modules<'py>(&self, py: Python<'py>) -> PyResult<Vec<String>> {
        let modules = self.inner.call_method0(py, "named_modules")?;
        let mut out = Vec::new();
        for item in modules.bind(py).try_iter()? {
            let name: String = item?.get_item(0)?.extract()?;
            if !name.is_empty() {
                out.push(name);
            }
        }
        Ok(out)
    }

    /// The model's Python class name, e.g. `Qwen3_5ForConditionalGeneration`.
    pub fn class_name(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.inner.bind(py).get_type().name()?.to_string())
    }
}

/// Make an embedded interpreter see the active virtualenv.
///
/// When Rust embeds Python (`cargo test`, the CLI), the interpreter starts
/// from its base install and does not see a venv's site-packages, so
/// `import transformers` fails even though the venv has it. When
/// `VIRTUAL_ENV` is set, add that venv's site-packages. `site.addsitedir`
/// ignores a directory that is already on the path, so a wheel loaded from
/// inside the venv is unaffected.
fn add_virtualenv_site(py: Python<'_>) -> PyResult<()> {
    let Ok(venv) = std::env::var("VIRTUAL_ENV") else {
        return Ok(());
    };
    let version = py.version_info();
    let site_packages = Path::new(&venv)
        .join("lib")
        .join(format!("python{}.{}", version.major, version.minor))
        .join("site-packages");
    py.import("site")?.call_method1(
        "addsitedir",
        (site_packages.to_string_lossy().into_owned(),),
    )?;
    Ok(())
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
    add_virtualenv_site(py)?;
    let transformers = py.import("transformers")?;
    let auto_model = transformers.getattr("AutoModel")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", "bfloat16")?;
    let model = auto_model.call_method("from_pretrained", (path,), Some(&kwargs))?;
    Ok(PyModel {
        inner: model.unbind(),
    })
}

#[pymodule]
fn sememe_bridge(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
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
        let (root, names) = Python::with_gil(|py| {
            let m = self.model.borrow(py);
            let root = m.class_name(py)?;
            Ok::<_, PyErr>((root, m.named_modules(py)?))
        })
        .map_err(|e| CoreError::Backend(e.to_string()))?;
        let children: Vec<ModulePath> = names.into_iter().map(ModulePath::new).collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// A model whose module walk fails partway. The old loop read any error
    /// as the end of the tree and returned the partial list.
    #[test]
    fn a_failing_module_walk_is_an_error_not_a_short_tree() {
        Python::with_gil(|py| {
            let code = CString::new(
                "class Broken:\n    def named_modules(self):\n        yield ('', None)\n        yield ('encoder', None)\n        raise RuntimeError('walk failed')\n",
            )
            .unwrap();
            let module = PyModule::from_code(py, &code, c"broken.py", c"broken").unwrap();
            let inner = module.getattr("Broken").unwrap().call0().unwrap().unbind();
            let model = PyModel { inner };
            let err = model
                .named_modules(py)
                .expect_err("a walk that raises must not succeed");
            assert!(err.to_string().contains("walk failed"), "{err}");
            assert_eq!(model.class_name(py).unwrap(), "Broken");
        });
    }
}
