use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use sememe::backend::stub::StubBackend;
use sememe::{
    EditOp, Harness, HubEvent, ModelInput, ModulePath, ModuleTree, TensorDType, TensorView,
};

fn build_stub() -> StubBackend {
    let tree = ModuleTree {
        root: "model".into(),
        children: vec![
            ModulePath::new("encoder"),
            ModulePath::new("encoder.layer.0"),
            ModulePath::new("encoder.layer.0.attention"),
        ],
    };
    let mut tensors = BTreeMap::new();
    tensors.insert(
        ModulePath::new("encoder.layer.0.attention"),
        vec![1.0_f32, 2.0, 3.0, 4.0],
    );
    StubBackend::new(tree, tensors)
}

#[test]
fn topology_returns_tree_and_records_it() {
    let mut h = Harness::new(build_stub());
    let tree = h.topology().unwrap();
    assert_eq!(tree.children.len(), 3);
    assert!(
        h.telemetry()
            .node_at(&ModulePath::new("encoder.layer.0"))
            .is_some()
    );
}

#[test]
fn forward_records_views_and_bumps_counter() {
    let mut h = Harness::new(build_stub());
    h.topology().unwrap();
    let views = h.forward(&ModelInput::from_ids(vec![1, 2])).unwrap();

    assert_eq!(views.len(), 1);
    let view = &views[0];
    assert_eq!(view.path, ModulePath::new("encoder.layer.0.attention"));
    assert_eq!(view.mean, 2.5);
    assert_eq!(h.telemetry().current_forward(), 1);

    // second forward bumps again, latest reflects new mean
    let _ = h.forward(&ModelInput::from_ids(vec![3, 4])).unwrap();
    let node = h
        .telemetry()
        .node_at(&ModulePath::new("encoder.layer.0.attention"))
        .unwrap();
    assert_eq!(node.history.len(), 2);
    assert_eq!(node.latest.as_ref().unwrap().forward_index, 2);
}

#[test]
fn edit_zero_wipes_stored_vector() {
    let mut h = Harness::new(build_stub());
    h.topology().unwrap();
    let before = h.forward(&ModelInput::from_ids(vec![1])).unwrap();
    let pre = before[0].mean;

    h.edit(&ModulePath::new("encoder.layer.0.attention"), &EditOp::Zero)
        .unwrap();

    let after = h.forward(&ModelInput::from_ids(vec![1])).unwrap();
    assert_eq!(after[0].mean, 0.0);
    assert_ne!(pre, 0.0);

    let node = h
        .telemetry()
        .node_at(&ModulePath::new("encoder.layer.0.attention"))
        .unwrap();
    assert_eq!(node.edits, vec![EditOp::Zero]);
}

#[test]
fn edit_then_forward_shows_diff() {
    let mut h = Harness::new(build_stub());
    h.topology().unwrap();
    let before = h.forward(&ModelInput::from_ids(vec![1])).unwrap();
    let before_mean = before[0].mean;

    h.edit(
        &ModulePath::new("encoder.layer.0.attention"),
        &EditOp::Scale(2.0),
    )
    .unwrap();

    let after = h.forward(&ModelInput::from_ids(vec![1])).unwrap();
    assert!((after[0].mean - before_mean * 2.0).abs() < 1e-6);
}

#[test]
fn subscriber_sees_topology_then_observation() {
    let mut h = Harness::new(build_stub());
    let events: Arc<Mutex<Vec<HubEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_cb = Arc::clone(&events);
    let _id = h
        .telemetry()
        .subscribe(move |e| events_cb.lock().unwrap().push(e.clone()));

    h.topology().unwrap();
    h.forward(&ModelInput::from_ids(vec![1])).unwrap();

    let events = events.lock().unwrap().clone();
    assert!(matches!(events[0], HubEvent::Topology { .. }));
    assert!(matches!(events[1], HubEvent::Observation { .. }));
}

#[test]
fn edit_bad_path_returns_invalid_path_error() {
    let mut h = Harness::new(build_stub());
    h.topology().unwrap();
    let err = h
        .edit(&ModulePath::new("does.not.exist"), &EditOp::Zero)
        .unwrap_err();
    assert!(matches!(err, sememe::Error::InvalidPath(_)));
}

#[test]
fn raw_constructors_smoke() {
    // sanity check that the public surface is reachable as documented
    let _path = ModulePath::new("a");
    let _view = TensorView::from_slice(ModulePath::new("a"), vec![1], TensorDType::Float32, &[0.0]);
    let _input = ModelInput::from_ids(vec![1, 2, 3]);
}
