//! PyO3 bridge to a live PyTorch model.
//!
//! M2 ships `load` and `named_modules`. M3 adds `run_forward` (hook
//! firing, Arrow IPC over the boundary). M5 adds `edit`. The bridge owns
//! the FFI boundary; the `sememe` core stays pure Rust.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array, ListArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PyModuleMethods, PyTuple};

use sememe::backend::Backend;
use sememe::{
    EditOp, Error as CoreError, ModelInput, ModulePath, ModuleTree, Result as CoreResult,
    TensorView, dtype::TensorDType,
};

type ArrayRef = Arc<dyn Array>;

/// Run a single forward pass with hooks on every named module. Returns
/// the Arrow IPC stream with one row per hook fire.
#[pyfunction]
#[pyo3(name = "run_forward_with_hooks")]
pub fn run_forward_with_hooks(
    py: Python<'_>,
    model_obj: &Bound<'_, PyAny>,
    ids: Vec<i64>,
    paths: Vec<String>,
) -> PyResult<Vec<u8>> {
    // Unwrap a `PyModel` wrapper if the function was passed one; the
    // raw transformers model is what carries `named_modules` and
    // `forward`.
    let raw = if let Ok(wrapper) = model_obj.extract::<Py<PyModel>>() {
        wrapper.borrow(py).inner.clone_ref(py)
    } else {
        model_obj.clone().unbind()
    };
    let raw_bound = raw.bind(py);

    // Walk each path through the model tree, collecting module handles.
    let mut hookables: Vec<(String, PyObject)> = Vec::with_capacity(paths.len());
    for name in &paths {
        let mut current_obj: PyObject = raw.clone_ref(py);
        let mut owner: Option<PyObject> = None;
        let mut ok = true;
        for seg in name.split('.') {
            let bound = current_obj.bind(py);
            match bound.getattr(seg) {
                Ok(child) => {
                    let child_obj = child.unbind();
                    owner = Some(child_obj.clone_ref(py));
                    current_obj = child_obj;
                }
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        if let Some(m) = owner {
            let bound = m.bind(py);
            if bound.hasattr("register_forward_hook")? {
                hookables.push((name.clone(), m));
            }
        }
    }

    let captured: Bound<'_, PyList> = PyList::empty(py);

    // Stamp each module with its dotted name so the hook can recover it.
    for (name, m) in &hookables {
        m.bind(py).setattr("__sememe_path__", name)?;
    }

    let functools = py.import("functools")?;
    let sememe_bridge = py.import("sememe_bridge")?;
    let hook_fn = sememe_bridge.getattr("_native")?.getattr("_capture_hook")?;
    let partial = functools.call_method("partial", (hook_fn, captured.clone()), None)?;

    // Register hooks.
    let mut hook_handles: Vec<PyObject> = Vec::with_capacity(hookables.len());
    for (_n, m) in &hookables {
        if let Ok(h) = m
            .bind(py)
            .call_method("register_forward_hook", (partial.clone(),), None)
        {
            hook_handles.push(h.unbind());
        }
    }

    // Run the forward.
    let forward_result = {
        let torch = py.import("torch")?;
        let ids_list = PyList::new(py, ids.iter().copied())?;
        let kw_dtype = PyDict::new(py);
        kw_dtype.set_item("dtype", torch.getattr("long")?)?;
        let input_ids = torch
            .call_method("tensor", (ids_list,), Some(&kw_dtype))?
            .call_method("unsqueeze", (0_i64,), None)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("input_ids", input_ids)?;
        raw_bound.call_method("forward", (), Some(&kwargs))
    };

    for h in &hook_handles {
        let _ = h.bind(py).call_method0("remove");
    }

    forward_result.map_err(|e| PyException::new_err(format!("forward failed: {e}")))?;

    let rows = captured.len();
    if rows == 0 {
        return Ok(empty_batch_bytes());
    }

    let mut path_col: Vec<String> = Vec::with_capacity(rows);
    let mut shape_col: Vec<Vec<i64>> = Vec::with_capacity(rows);
    let mut dtype_col: Vec<String> = Vec::with_capacity(rows);
    let mut min_col: Vec<f64> = Vec::with_capacity(rows);
    let mut max_col: Vec<f64> = Vec::with_capacity(rows);
    let mut mean_col: Vec<f64> = Vec::with_capacity(rows);
    let mut samples_col: Vec<Vec<f64>> = Vec::with_capacity(rows);

    for i in 0..rows {
        let item = captured.get_item(i)?;
        let dict: &Bound<'_, PyDict> = item.downcast()?;
        let path: String = dict.get_item("path")?.unwrap().extract()?;
        let shape: Vec<i64> = dict.get_item("shape")?.unwrap().extract()?;
        let dtype: String = dict.get_item("dtype")?.unwrap().extract()?;
        let min: f64 = dict.get_item("min")?.unwrap().extract()?;
        let max: f64 = dict.get_item("max")?.unwrap().extract()?;
        let mean: f64 = dict.get_item("mean")?.unwrap().extract()?;
        let samples: Vec<f64> = dict.get_item("samples")?.unwrap().extract()?;
        path_col.push(path);
        shape_col.push(shape);
        dtype_col.push(dtype);
        min_col.push(min);
        max_col.push(max);
        mean_col.push(mean);
        samples_col.push(samples);
    }

    encode_batch(&ArrowColumns {
        paths: &path_col,
        shapes: &shape_col,
        dtypes: &dtype_col,
        mins: &min_col,
        maxs: &max_col,
        means: &mean_col,
        samples: &samples_col,
    })
    .map_err(|e| PyException::new_err(format!("encode: {e}")))
}

/// Free-function counterpart to `PyModel.named_modules`. Accepts
/// either a wrapped `PyModel` or a raw transformers model.
#[pyfunction]
#[pyo3(name = "named_modules")]
pub fn named_modules(py: Python<'_>, model_obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let raw = if let Ok(wrapper) = model_obj.extract::<Py<PyModel>>() {
        wrapper.borrow(py).inner.clone_ref(py)
    } else {
        model_obj.clone().unbind()
    };
    let iter = raw.call_method0(py, "named_modules")?;
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

#[pyclass]
pub struct PyModel {
    inner: PyObject,
}

#[pymethods]
impl PyModel {
    #[pyo3(name = "named_modules")]
    pub fn named_modules_py(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        named_modules(py, self.inner.bind(py))
    }
}

/// Load a PyTorch model from a HuggingFace cache snapshot directory.
#[pyfunction]
#[pyo3(name = "load")]
pub fn load(py: Python<'_>, path: &str) -> PyResult<Py<PyModel>> {
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
    Py::new(
        py,
        PyModel {
            inner: model.unbind(),
        },
    )
}

/// Return the Qwen snapshot path the integration tests use.
#[pyfunction]
#[pyo3(name = "demo_model_path")]
pub fn demo_model_path() -> &'static str {
    "/Users/ericmey/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
}

/// Hook callable used by `run_forward_with_hooks`.
#[pyfunction]
#[pyo3(name = "_capture_hook")]
pub fn capture_hook(
    py: Python<'_>,
    captured: &Bound<'_, PyList>,
    module: &Bound<'_, PyAny>,
    _input: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let tensor = first_tensor(output)?;

    let torch = py.import("torch")?;
    let tensor_type = torch.getattr("Tensor")?;
    if !tensor.is_instance(&tensor_type)? {
        return Ok(());
    }

    let t = tensor.call_method0("detach")?.call_method0("cpu")?;
    let size_obj = t.call_method0("size")?;
    let shape: Vec<i64> = size_obj
        .try_iter()?
        .map(|i| i.and_then(|x| x.extract::<i64>()))
        .collect::<PyResult<Vec<_>>>()?;
    let dtype_obj = t.getattr("dtype")?.str()?.to_string();

    let flat = t.call_method("flatten", (), None)?;
    let casted = flat.call_method("to", (torch.getattr("float32")?,), None)?;
    let values: Vec<f64> = casted.call_method0("tolist")?.extract()?;

    let (mn, mx, mean) = if values.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let mn = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        (mn, mx, mean)
    };
    let samples: Vec<f64> = values.iter().take(8).copied().collect();

    let path: String = module
        .getattr("__sememe_path__")
        .and_then(|v| v.extract())
        .unwrap_or_default();

    let dict = PyDict::new(py);
    dict.set_item("path", path)?;
    dict.set_item("shape", shape)?;
    dict.set_item("dtype", dtype_obj)?;
    dict.set_item("min", mn)?;
    dict.set_item("max", mx)?;
    dict.set_item("mean", mean)?;
    dict.set_item("samples", samples)?;
    captured.append(dict)?;
    Ok(())
}

fn first_tensor<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        if let Some(first) = tuple.iter().next() {
            return Ok(first);
        }
    }
    Ok(obj.clone())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(named_modules, m)?)?;
    m.add_function(wrap_pyfunction!(run_forward_with_hooks, m)?)?;
    m.add_function(wrap_pyfunction!(demo_model_path, m)?)?;
    m.add_function(wrap_pyfunction!(capture_hook, m)?)?;
    m.add_class::<PyModel>()?;
    Ok(())
}

// ---- Rust-side typed handle -----------------------------------------------

pub struct PyBackend {
    model: Py<PyModel>,
}

impl PyBackend {
    pub fn load(path: &str) -> CoreResult<Self> {
        Python::with_gil(|py| {
            let model = load(py, path).map_err(|e| CoreError::Backend(e.to_string()))?;
            Ok(Self { model })
        })
    }
}

impl Backend for PyBackend {
    fn named_modules(&self) -> CoreResult<ModuleTree> {
        let names: Vec<String> = Python::with_gil(|py| {
            self.model
                .borrow(py)
                .named_modules_py(py)
                .map_err(|e| CoreError::Backend(e.to_string()))
        })?;
        let children: Vec<ModulePath> = names.into_iter().map(ModulePath::new).collect();
        let root = children
            .first()
            .map(|p| p.as_str().split('.').next().unwrap_or("").to_string())
            .unwrap_or_default();
        Ok(ModuleTree { root, children })
    }

    fn run_forward(&self, input: &ModelInput) -> CoreResult<Vec<TensorView>> {
        let tree = self.named_modules()?;
        let paths: Vec<String> = tree
            .children
            .iter()
            .map(|p| p.as_str().to_string())
            .collect();
        let ids = input.ids.clone();
        let bytes: Vec<u8> = Python::with_gil(|py| {
            let bound = self.model.borrow(py);
            let inner_bound = bound.inner.bind(py);
            run_forward_with_hooks(py, inner_bound, ids, paths)
                .map_err(|e| CoreError::Backend(e.to_string()))
        })?;
        decode_views(&bytes)
    }

    fn edit(&mut self, _path: &ModulePath, _op: &EditOp) -> CoreResult<()> {
        Err(CoreError::Unsupported(
            "Backend::edit is not implemented; lands in M5",
        ))
    }
}

// ---- Arrow encoding / decoding --------------------------------------------

fn arrow_schema() -> Schema {
    Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new(
            "shape",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        ),
        Field::new("dtype", DataType::Utf8, false),
        Field::new("min", DataType::Float64, false),
        Field::new("max", DataType::Float64, false),
        Field::new("mean", DataType::Float64, false),
        Field::new(
            "samples",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
    ])
}

struct ArrowColumns<'a> {
    paths: &'a [String],
    shapes: &'a [Vec<i64>],
    dtypes: &'a [String],
    mins: &'a [f64],
    maxs: &'a [f64],
    means: &'a [f64],
    samples: &'a [Vec<f64>],
}

fn encode_batch(cols: &ArrowColumns<'_>) -> Result<Vec<u8>, String> {
    let schema = Arc::new(arrow_schema());
    let rows = cols.paths.len();

    let path_arr: ArrayRef = Arc::new(StringArray::from(cols.paths.to_vec()));

    // shape list<i64>
    let mut shape_offsets: Vec<i32> = Vec::with_capacity(rows + 1);
    let mut shape_values: Vec<i64> = Vec::new();
    shape_offsets.push(0);
    for s in cols.shapes {
        shape_values.extend(s.iter().copied());
        shape_offsets.push(shape_values.len() as i32);
    }
    let shape_field = Arc::new(Field::new("item", DataType::Int64, true));
    let shape_arr: ArrayRef = Arc::new(
        ListArray::try_new(
            shape_field,
            arrow::buffer::OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(shape_offsets)),
            Arc::new(Int64Array::from(shape_values)),
            None,
        )
        .map_err(|e| format!("shape list: {e}"))?,
    );

    let dtype_arr: ArrayRef = Arc::new(StringArray::from(cols.dtypes.to_vec()));

    let min_arr: ArrayRef = Arc::new(Float64Array::from(cols.mins.to_vec()));
    let max_arr: ArrayRef = Arc::new(Float64Array::from(cols.maxs.to_vec()));
    let mean_arr: ArrayRef = Arc::new(Float64Array::from(cols.means.to_vec()));

    // samples list<f64>
    let mut samples_offsets: Vec<i32> = Vec::with_capacity(rows + 1);
    let mut samples_values: Vec<f64> = Vec::new();
    samples_offsets.push(0);
    for s in cols.samples {
        samples_values.extend(s.iter().copied());
        samples_offsets.push(samples_values.len() as i32);
    }
    let samples_field = Arc::new(Field::new("item", DataType::Float64, true));
    let samples_arr: ArrayRef = Arc::new(
        ListArray::try_new(
            samples_field,
            arrow::buffer::OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(samples_offsets)),
            Arc::new(Float64Array::from(samples_values)),
            None,
        )
        .map_err(|e| format!("samples list: {e}"))?,
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            path_arr,
            shape_arr,
            dtype_arr,
            min_arr,
            max_arr,
            mean_arr,
            samples_arr,
        ],
    )
    .map_err(|e| format!("record batch: {e}"))?;

    let mut buf = Vec::new();
    {
        let mut writer =
            StreamWriter::try_new(&mut buf, &schema).map_err(|e| format!("writer: {e}"))?;
        writer
            .write(&batch)
            .map_err(|e| format!("writer write: {e}"))?;
        writer.finish().map_err(|e| format!("writer finish: {e}"))?;
    }
    Ok(buf)
}

fn empty_batch_bytes() -> Vec<u8> {
    let schema = Arc::new(arrow_schema());
    let batch = RecordBatch::new_empty(schema.clone());
    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &schema).expect("empty writer");
    writer.write(&batch).expect("empty write");
    writer.finish().expect("empty finish");
    buf
}

fn decode_views(bytes: &[u8]) -> CoreResult<Vec<TensorView>> {
    let expected = arrow_schema();
    let cursor = std::io::Cursor::new(bytes);
    let reader = StreamReader::try_new_buffered(cursor, None)
        .map_err(|e| CoreError::Backend(format!("arrow stream: {e}")))?;
    let batches: Vec<RecordBatch> = reader
        .collect::<Result<_, _>>()
        .map_err(|e| CoreError::Backend(format!("arrow read: {e}")))?;

    let mut out = Vec::new();
    for batch in batches {
        let schema = batch.schema();
        if schema.as_ref() != &expected {
            return Err(CoreError::Backend(format!(
                "arrow schema mismatch: got {:?}, expected {:?}",
                schema, expected
            )));
        }

        let paths = batch
            .column_by_name("path")
            .and_then(|a| a.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| CoreError::Backend("arrow: path column missing".into()))?;
        let shapes = batch
            .column_by_name("shape")
            .and_then(|a| a.as_any().downcast_ref::<ListArray>())
            .ok_or_else(|| CoreError::Backend("arrow: shape column missing".into()))?;
        let dtypes = batch
            .column_by_name("dtype")
            .and_then(|a| a.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| CoreError::Backend("arrow: dtype column missing".into()))?;
        let mins = batch
            .column_by_name("min")
            .and_then(|a| a.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| CoreError::Backend("arrow: min column missing".into()))?;
        let maxs = batch
            .column_by_name("max")
            .and_then(|a| a.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| CoreError::Backend("arrow: max column missing".into()))?;
        let means = batch
            .column_by_name("mean")
            .and_then(|a| a.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| CoreError::Backend("arrow: mean column missing".into()))?;
        let samples = batch
            .column_by_name("samples")
            .and_then(|a| a.as_any().downcast_ref::<ListArray>())
            .ok_or_else(|| CoreError::Backend("arrow: samples column missing".into()))?;

        for row in 0..batch.num_rows() {
            let path = paths.value(row).to_string();
            let shape_arr = shapes.value(row);
            let shape_inner = shape_arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| CoreError::Backend("arrow: shape.inner not i64".into()))?;
            let shape: Vec<usize> = (0..shape_inner.len())
                .map(|i| shape_inner.value(i).max(0) as usize)
                .collect();

            let dtype = match dtypes.value(row) {
                "torch.float32" | "f32" => TensorDType::Float32,
                "torch.float16" | "f16" => TensorDType::Float16,
                "torch.bfloat16" | "bf16" => TensorDType::Bfloat16,
                "torch.int64" | "i64" => TensorDType::Int64,
                "torch.bool" | "bool" => TensorDType::Bool,
                other => {
                    return Err(CoreError::Backend(format!(
                        "arrow: unknown dtype '{other}'"
                    )));
                }
            };
            let min = mins.value(row);
            let max = maxs.value(row);
            let mean = means.value(row);

            let samples_arr = samples.value(row);
            let samples_inner = samples_arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| CoreError::Backend("arrow: samples.inner not f64".into()))?;
            let samples_vec: Vec<f64> = (0..samples_inner.len())
                .map(|i| samples_inner.value(i))
                .collect();

            let view = TensorView {
                path: ModulePath::new(path),
                shape,
                dtype,
                min,
                max,
                mean,
                samples: samples_vec,
                forward_index: 0,
            };
            out.push(view);
        }
    }
    Ok(out)
}
