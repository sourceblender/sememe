//! End-to-end smoke test: load a real model and walk its module tree
//! through `PyBackend`.
//!
//! These tests never skip. A missing model or missing Python dependencies
//! is a failure, because a test that passes without loading anything
//! proves nothing. `just test` sets `VIRTUAL_ENV` to the project venv and
//! `SEMEME_TEST_MODEL` to the cached Qwen3.5-0.8B snapshot; set
//! `SEMEME_TEST_MODEL` yourself to try another model.

use sememe::backend::Backend;
use sememe_bridge::PyBackend;

fn model_path() -> String {
    std::env::var("SEMEME_TEST_MODEL")
        .expect("SEMEME_TEST_MODEL must name a model snapshot directory; `just test` sets it")
}

fn load() -> PyBackend {
    let path = model_path();
    PyBackend::load(&path).unwrap_or_else(|e| {
        panic!("PyBackend::load({path}) failed; is VIRTUAL_ENV set to a venv with torch and transformers? {e}")
    })
}

#[test]
fn load_and_walk_qwen_module_tree() {
    let tree = load().named_modules().expect("named_modules");
    assert!(!tree.children.is_empty(), "expected non-empty tree");
    // The root names the model, not its first child.
    assert!(
        tree.root.starts_with("Qwen"),
        "root should be the model's class name; got {:?}",
        tree.root
    );

    // Qwen3.5-0.8B has a `visual` (vision tower) and `language_model`
    // (LLM half). The harness is for LLMs; both branches are valid
    // targets but `language_model` is what we'll navigate into.
    let names: Vec<&str> = tree.children.iter().map(|p| p.as_str()).collect();
    assert!(
        names.iter().any(|n| n.starts_with("language_model")),
        "expected a `language_model` branch; got first 5: {:?}",
        &names[..names.len().min(5)]
    );
    assert!(
        names.iter().any(|n| n.starts_with("visual")),
        "expected a `visual` branch; got first 5: {:?}",
        &names[..names.len().min(5)]
    );
}

#[test]
fn unsupported_methods_surface_correctly() {
    let err = load()
        .run_forward(&sememe::ModelInput::from_ids(vec![1]))
        .expect_err("run_forward should fail in M2");
    assert!(matches!(err, sememe::Error::Unsupported(_)));
}
