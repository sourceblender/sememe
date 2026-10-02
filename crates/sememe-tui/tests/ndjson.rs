use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use sememe::HubEvent;
use sememe::{EditOp, ModulePath, TensorDType, TensorView};
use sememe_tui::ndjson::{NdjsonSink, sink_subscriber};

fn tmpfile(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sememe-tui-test-{name}.ndjson"));
    let _ = std::fs::remove_file(&p);
    p
}

fn view() -> TensorView {
    TensorView::from_slice(
        ModulePath::new("a"),
        vec![2],
        TensorDType::Float32,
        &[1.0, 2.0],
    )
}

#[test]
fn sink_writes_one_ndjson_line_per_event() {
    let path = tmpfile("writes");
    let sink = std::sync::Arc::new(NdjsonSink::open(&path).unwrap());
    let counter = Arc::new(AtomicU64::new(0));
    let counter_cb = Arc::clone(&counter);
    let clock: Arc<dyn Fn() -> u64 + Send + Sync> =
        Arc::new(move || counter_cb.load(Ordering::SeqCst));
    let subscriber = sink_subscriber(sink, clock);

    subscriber(&HubEvent::Topology {
        tree: sememe::ModuleTree {
            root: "model".into(),
            children: vec![ModulePath::new("a")],
        },
    });
    subscriber(&HubEvent::Observation {
        path: ModulePath::new("a"),
        view: view(),
    });
    subscriber(&HubEvent::Edit {
        path: ModulePath::new("a"),
        op: EditOp::Zero,
        before: None,
        after: Box::new(None),
    });

    let contents = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = contents.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("\"Topology\""));
    assert!(lines[1].contains("\"Observation\""));
    assert!(lines[2].contains("\"Edit\""));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn sink_records_round_trip_through_serde() {
    use sememe::RecordedEvent;

    let path = tmpfile("serde");
    let sink = std::sync::Arc::new(NdjsonSink::open(&path).unwrap());
    let clock: Arc<dyn Fn() -> u64 + Send + Sync> = Arc::new(|| 42);
    let subscriber = sink_subscriber(sink, clock);

    subscriber(&HubEvent::Observation {
        path: ModulePath::new("a"),
        view: view(),
    });

    let contents = std::fs::read_to_string(&path).unwrap();
    let parsed: RecordedEvent = serde_json::from_str(contents.trim()).unwrap();
    match parsed {
        RecordedEvent::Observation { view, recorded_at } => {
            assert_eq!(view.path.as_str(), "a");
            assert_eq!(recorded_at, 42);
        }
        _ => panic!("expected Observation"),
    }

    let _ = std::fs::remove_file(&path);
}
