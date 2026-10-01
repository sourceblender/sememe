use std::sync::{Arc, Mutex};

use sememe::{
    EditOp, HubEvent, ModulePath, ModuleTree, SubscriptionId, Telemetry, TensorDType, TensorView,
};

fn view(path: &str, values: &[f32]) -> TensorView {
    TensorView::from_slice(
        ModulePath::new(path),
        vec![values.len()],
        TensorDType::Float32,
        values,
    )
}

#[test]
fn begin_forward_increments_counter() {
    let mut t = Telemetry::new();
    assert_eq!(t.current_forward(), 0);
    let i = t.begin_forward();
    assert_eq!(i, 1);
    assert_eq!(t.current_forward(), 1);
    let j = t.begin_forward();
    assert_eq!(j, 2);
}

#[test]
fn observation_creates_path_and_records_latest() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("encoder.layer.0", &[1.0, 2.0, 3.0]));

    let node = t.node_at(&ModulePath::new("encoder.layer.0")).unwrap();
    let latest = node.latest.as_ref().unwrap();
    assert_eq!(latest.min, 1.0);
    assert_eq!(latest.max, 3.0);
    assert_eq!(latest.mean, 2.0);
}

#[test]
fn observation_creates_intermediate_nodes() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("encoder.layer.0.attention", &[0.5]));

    assert!(t.node_at(&ModulePath::new("encoder")).is_some());
    assert!(t.node_at(&ModulePath::new("encoder.layer")).is_some());
    assert!(t.node_at(&ModulePath::new("encoder.layer.0")).is_some());
    assert!(
        t.node_at(&ModulePath::new("encoder.layer.0.attention"))
            .is_some()
    );
    assert!(t.node_at(&ModulePath::new("classifier")).is_none());
}

#[test]
fn iter_visits_every_node_in_preorder() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("a.b", &[1.0]));
    t.record_observation(view("a.c", &[2.0]));

    let names: Vec<String> = t.iter().map(|n| n.path().to_string()).collect();
    assert_eq!(names, vec!["", "a", "a.b", "a.c"]);
}

#[test]
fn history_is_bounded() {
    let mut t = Telemetry::with_history_cap(2);
    for i in 1..=5 {
        t.begin_forward();
        t.record_observation(view("a", &[i as f32]));
    }
    let node = t.node_at(&ModulePath::new("a")).unwrap();
    assert_eq!(node.history.len(), 2);
    assert_eq!(node.history[0].mean, 4.0);
    assert_eq!(node.history[1].mean, 5.0);
    assert_eq!(node.latest.as_ref().unwrap().mean, 5.0);
}

#[test]
fn forwards_since_last_obs() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("a", &[1.0]));
    t.begin_forward();
    t.begin_forward();
    let node = t.node_at(&ModulePath::new("a")).unwrap();
    assert_eq!(node.forwards_since_last_obs(t.current_forward()), 2);
}

#[test]
fn never_observed_node_does_not_exist() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.begin_forward();
    assert!(t.node_at(&ModulePath::new("missing")).is_none());
}

#[test]
fn last_mean_delta() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("a", &[1.0]));
    assert_eq!(
        t.node_at(&ModulePath::new("a")).unwrap().last_mean_delta(),
        None
    );
    t.begin_forward();
    t.record_observation(view("a", &[3.0]));
    assert_eq!(
        t.node_at(&ModulePath::new("a")).unwrap().last_mean_delta(),
        Some(2.0)
    );
}

#[test]
fn record_edit_records_path_and_op() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("a", &[5.0]));
    t.record_edit(&ModulePath::new("a"), EditOp::Zero);
    let node = t.node_at(&ModulePath::new("a")).unwrap();
    assert_eq!(node.edits, vec![EditOp::Zero]);
}

#[test]
fn subscriber_fires_on_observation() {
    let mut t = Telemetry::new();
    let seen: Arc<Mutex<Vec<ModulePath>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_cb = Arc::clone(&seen);
    let _id: SubscriptionId = t.subscribe(move |event| {
        if let HubEvent::Observation { path, .. } = event {
            seen_cb.lock().unwrap().push(path.clone());
        }
    });
    t.begin_forward();
    t.record_observation(view("a", &[1.0]));
    t.record_observation(view("b", &[2.0]));
    let paths = seen.lock().unwrap().clone();
    assert_eq!(paths, vec![ModulePath::new("a"), ModulePath::new("b")]);
}

#[test]
fn subscriber_fires_on_edit_with_before_captured() {
    let mut t = Telemetry::new();
    let seen: Arc<Mutex<Option<HubEvent>>> = Arc::new(Mutex::new(None));
    let seen_cb = Arc::clone(&seen);
    t.subscribe(move |event| {
        if let HubEvent::Edit { .. } = event {
            *seen_cb.lock().unwrap() = Some(event.clone());
        }
    });
    t.begin_forward();
    t.record_observation(view("a", &[5.0]));
    t.record_edit(&ModulePath::new("a"), EditOp::Zero);

    let event = seen.lock().unwrap().clone().expect("event fired");
    match event {
        HubEvent::Edit {
            path,
            op,
            before,
            after,
        } => {
            assert_eq!(path, ModulePath::new("a"));
            assert_eq!(op, EditOp::Zero);
            assert!(before.is_some());
            assert!(after.is_none());
        }
        _ => unreachable!(),
    }
}

#[test]
fn unsubscribe_stops_callbacks() {
    let mut t = Telemetry::new();
    let count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let count_cb = Arc::clone(&count);
    let id = t.subscribe(move |_| *count_cb.lock().unwrap() += 1);
    t.begin_forward();
    t.record_observation(view("a", &[1.0]));
    t.unsubscribe(id);
    t.begin_forward();
    t.record_observation(view("a", &[2.0]));
    assert_eq!(*count.lock().unwrap(), 1);
}

#[test]
fn topology_replaces_tree_and_notifies() {
    let mut t = Telemetry::new();
    t.begin_forward();
    t.record_observation(view("stale.path", &[1.0]));

    let seen: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let seen_cb = Arc::clone(&seen);
    t.subscribe(move |event| {
        if matches!(event, HubEvent::Topology { .. }) {
            *seen_cb.lock().unwrap() = true;
        }
    });

    let tree = ModuleTree {
        root: "model".into(),
        children: vec![ModulePath::new("encoder"), ModulePath::new("classifier")],
    };
    t.record_topology(&tree).unwrap();

    assert!(*seen.lock().unwrap());
    assert!(t.node_at(&ModulePath::new("encoder")).is_some());
    assert!(t.node_at(&ModulePath::new("classifier")).is_some());
    assert!(t.node_at(&ModulePath::new("stale.path")).is_none());
}
