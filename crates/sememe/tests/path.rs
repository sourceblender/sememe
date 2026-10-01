use sememe::ModulePath;

#[test]
fn segments_splits_on_dots() {
    let p = ModulePath::new("encoder.layer.3.attention");
    let segs: Vec<&str> = p.segments().collect();
    assert_eq!(segs, vec!["encoder", "layer", "3", "attention"]);
}

#[test]
fn empty_segments_skipped() {
    let p = ModulePath::new(".encoder..layer.");
    let segs: Vec<&str> = p.segments().collect();
    assert_eq!(segs, vec!["encoder", "layer"]);
}

#[test]
fn serde_round_trip() {
    let p = ModulePath::new("a.b.c");
    let json = serde_json::to_string(&p).unwrap();
    let back: ModulePath = serde_json::from_str(&json).unwrap();
    assert_eq!(p, back);
}

#[test]
fn from_string_and_str() {
    let a: ModulePath = "x.y".into();
    let b: ModulePath = String::from("x.y").into();
    assert_eq!(a, b);
}
