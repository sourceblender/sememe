//! End-to-end smoke test: load the real Qwen3.5-0.8B from the HF cache
//! snapshot and walk its module tree through `PyBackend`. Run with
//! `cargo test -p sememe-bridge --test smoke` from a venv where
//! `transformers` and `torch` are installed.

use sememe::backend::Backend;
use sememe_bridge::PyBackend;

const DEMO_MODEL_PATH: &str = "/Users/ericmey/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

#[test]
fn load_and_walk_qwen_module_tree() {
    let backend = match PyBackend::load(DEMO_MODEL_PATH) {
        Ok(b) => b,
        Err(e) => {
            // Skip gracefully if transformers/torch aren't installed in
            // the active Python environment — keeps `cargo test --workspace`
            // green on machines without the venv set up.
            eprintln!("skipping: PyBackend::load failed (transformers/torch installed?): {e}");
            return;
        }
    };

    let tree = backend.named_modules().expect("named_modules");
    assert!(!tree.children.is_empty(), "expected non-empty tree");

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
    let backend = match PyBackend::load(DEMO_MODEL_PATH) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let err = backend
        .run_forward(&sememe::ModelInput::from_ids(vec![1]))
        .expect_err("run_forward should fail in M2");
    assert!(matches!(err, sememe::Error::Unsupported(_)));
}
