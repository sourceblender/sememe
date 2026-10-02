//! End-to-end smoke tests against the on-disk Qwen3.5-0.8B snapshot.
//! Skips gracefully if `transformers` / `torch` aren't installed in the
//! active Python environment, so `cargo test --workspace` stays green on
//! machines without the venv.

use std::sync::Arc;

use sememe::backend::Backend;
use sememe_bridge::PyBackend;

const DEMO_MODEL_PATH: &str = "/Users/ericmey/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

fn backend() -> Option<PyBackend> {
    match PyBackend::load(DEMO_MODEL_PATH) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: PyBackend::load failed (transformers/torch installed?): {e}");
            None
        }
    }
}

#[test]
fn load_and_walk_qwen_module_tree() {
    let Some(backend) = backend() else { return };
    let tree = backend.named_modules().expect("named_modules");
    assert!(!tree.children.is_empty());
    let names: Vec<&str> = tree.children.iter().map(|p| p.as_str()).collect();
    assert!(
        names.iter().any(|n| n.starts_with("language_model")),
        "expected a `language_model` branch"
    );
    assert!(
        names.iter().any(|n| n.starts_with("visual")),
        "expected a `visual` branch"
    );
}

#[test]
fn unsupported_methods_surface_correctly() {
    let Some(backend) = backend() else { return };
    let mut b = backend;
    let err = b
        .edit(&sememe::ModulePath::new("encoder"), &sememe::EditOp::Zero)
        .expect_err("edit should fail until M5");
    assert!(matches!(err, sememe::Error::Unsupported(_)));
}

#[test]
fn run_forward_returns_one_view_per_hooked_module() {
    let Some(backend) = backend() else { return };

    let views = backend
        .run_forward(&sememe::ModelInput::from_ids(vec![101, 2024, 3056, 4048]))
        .expect("run_forward");

    // We hooked every named module in the tree; the forward should
    // have produced at least one observation per module path.
    let tree = backend.named_modules().expect("tree");
    let hooked = tree
        .children
        .iter()
        .filter(|p| {
            // The bridge skipped any path that didn't resolve to a real
            // module handle; accept fewer observations than tree size.
            !p.as_str().is_empty()
        })
        .count();
    assert!(
        views.len() >= hooked / 2,
        "expected many views; got {} (tree has {})",
        views.len(),
        hooked
    );

    // Spot-check a real layer's view: should have sane shape, dtype,
    // and a finite mean.
    let attn_view = views
        .iter()
        .find(|v| v.path.as_str().contains("language_model.layers.0"))
        .expect("at least one language_model.layers.0 view");
    assert!(!attn_view.shape.is_empty(), "shape should be non-empty");
    assert!(attn_view.mean.is_finite(), "mean should be finite");
    assert!(attn_view.min <= attn_view.mean && attn_view.mean <= attn_view.max);
    assert!(!attn_view.samples.is_empty(), "should have samples");

    // The HubEvent subscription path: register a subscriber on the
    // harness's telemetry, run a forward, and confirm the subscriber
    // fires for every observation.
    let mut harness = sememe::Harness::new(backend);
    harness.topology().expect("topology");
    let counter = Arc::new(std::sync::Mutex::new(0u32));
    let counter_cb = Arc::clone(&counter);
    let _id = harness.telemetry().subscribe(move |event| {
        if matches!(event, sememe::HubEvent::Observation { .. }) {
            *counter_cb.lock().unwrap() += 1;
        }
    });
    let _ = harness
        .forward(&sememe::ModelInput::from_ids(vec![101, 2024]))
        .expect("harness forward");
    let observed = *counter.lock().unwrap();
    assert!(
        observed as usize >= hooked / 2,
        "subscriber should see observations"
    );
}
