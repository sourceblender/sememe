use sememe::{ModulePath, TensorDType, TensorView};

#[test]
fn from_slice_computes_min_max_mean() {
    let values = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let view = TensorView::from_slice(
        ModulePath::new("test"),
        vec![5],
        TensorDType::Float32,
        &values,
    );
    assert_eq!(view.min, 1.0);
    assert_eq!(view.max, 5.0);
    assert_eq!(view.mean, 3.0);
    assert_eq!(view.samples, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn from_slice_empty_input_has_zero_mean() {
    let view = TensorView::from_slice(ModulePath::new("test"), vec![0], TensorDType::Float32, &[]);
    assert_eq!(view.min, f64::INFINITY);
    assert_eq!(view.max, f64::NEG_INFINITY);
    assert_eq!(view.mean, 0.0);
}

#[test]
fn samples_capped_at_eight() {
    let values = (0..20).map(|i| i as f32).collect::<Vec<_>>();
    let view = TensorView::from_slice(
        ModulePath::new("test"),
        vec![20],
        TensorDType::Float32,
        &values,
    );
    assert_eq!(view.samples.len(), 8);
}

#[test]
fn serde_round_trip() {
    let view = TensorView::from_slice(
        ModulePath::new("a.b"),
        vec![3],
        TensorDType::Float32,
        &[0.5, 1.5, 2.5],
    );
    let json = serde_json::to_string(&view).unwrap();
    let back: TensorView = serde_json::from_str(&json).unwrap();
    assert_eq!(view, back);
}
