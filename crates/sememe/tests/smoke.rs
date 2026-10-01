#[test]
fn version_matches_manifest() {
    assert_eq!(sememe::version(), env!("CARGO_PKG_VERSION"));
}
