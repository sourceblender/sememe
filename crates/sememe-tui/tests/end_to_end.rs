//! End-to-end wiring: harness → telemetry → NDJSON sink. Verifies that
//! a subscriber receives events and writes them to disk. The ratatui
//! runtime itself isn't exercised here (no terminal), but the data
//! path the TUI reads from is.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use sememe::backend::stub::StubBackend;
use sememe::{EditOp, Harness, ModelInput, ModulePath, ModuleTree};
use sememe_tui::ndjson::{NdjsonSink, sink_subscriber};

fn tmpfile(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sememe-tui-e2e-{name}.ndjson"));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn ndjson_sink_writes_hub_events_during_harness_run() {
    let path = tmpfile("run");
    let sink = Arc::new(NdjsonSink::open(&path).unwrap());
    let clock: Arc<dyn Fn() -> u64 + Send + Sync> = Arc::new(|| 0);
    let sub = sink_subscriber(Arc::clone(&sink), Arc::clone(&clock));
    let _id = sememe::Telemetry::new().subscribe(sub); // ensure closure types check

    let mut tree = BTreeMap::new();
    tree.insert(ModulePath::new("a.b"), vec![1.0_f32, 2.0, 3.0]);
    let stub = StubBackend::new(
        ModuleTree {
            root: "model".into(),
            children: vec![ModulePath::new("a.b")],
        },
        tree,
    );

    let mut harness = Harness::new(stub);
    let sub = sink_subscriber(Arc::clone(&sink), Arc::clone(&clock));
    harness.telemetry().subscribe(sub);

    harness.topology().unwrap();
    let _ = harness.forward(&ModelInput::from_ids(vec![1])).unwrap();
    harness
        .edit(&ModulePath::new("a.b"), &EditOp::Zero)
        .unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    assert_eq!(lines.len(), 3, "expected 3 events; got: {contents}");
    assert!(lines[0].contains("\"Topology\""));
    assert!(lines[1].contains("\"Observation\""));
    assert!(lines[2].contains("\"Edit\""));

    let _ = std::fs::remove_file(&path);
    let _ = (sink, clock, _id);
}
