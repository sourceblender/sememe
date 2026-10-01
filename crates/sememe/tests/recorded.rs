use sememe::{EditOp, ModelInput, ModulePath, ModuleTree, RecordedEvent, TensorDType, TensorView};

#[test]
fn topology_round_trip() {
    let event = RecordedEvent::Topology {
        tree: ModuleTree {
            root: "model".into(),
            children: vec![ModulePath::new("encoder"), ModulePath::new("classifier")],
        },
        recorded_at: 1,
    };
    let json = serde_json::to_string(&event).unwrap();
    let back: RecordedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(event, back);
    assert!(json.contains("\"kind\":\"Topology\""));
}

#[test]
fn forward_round_trip() {
    let event = RecordedEvent::Forward {
        input: ModelInput::from_ids(vec![1, 2, 3]),
        views: vec![TensorView::from_slice(
            ModulePath::new("a"),
            vec![3],
            TensorDType::Float32,
            &[1.0, 2.0, 3.0],
        )],
        recorded_at: 2,
    };
    let json = serde_json::to_string(&event).unwrap();
    let back: RecordedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(event, back);
}

#[test]
fn observation_round_trip() {
    let event = RecordedEvent::Observation {
        view: TensorView::from_slice(ModulePath::new("a"), vec![1], TensorDType::Float32, &[42.0]),
        recorded_at: 3,
    };
    let json = serde_json::to_string(&event).unwrap();
    let back: RecordedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(event, back);
}

#[test]
fn edit_round_trip() {
    let event = RecordedEvent::Edit {
        path: ModulePath::new("encoder.layer.0"),
        op: EditOp::Scale(2.0),
        before: None,
        after: Box::new(None),
        recorded_at: 4,
    };
    let json = serde_json::to_string(&event).unwrap();
    let back: RecordedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(event, back);
}
