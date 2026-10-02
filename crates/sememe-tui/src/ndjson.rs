//! NDJSON session-log sink. A `HubEvent` subscriber that writes each
//! event to a file as a `RecordedEvent`. M4 wires it as a separate
//! subscriber on the harness's telemetry; M6 reads it back via
//! `Replayer`.
//!
//! The sink takes `std::sync::Mutex<std::fs::File>` so multiple
//! subscribers writing concurrently don't interleave.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

use sememe::{Error as CoreError, HubEvent, RecordedEvent, Result as CoreResult};

/// NDJSON session-log writer. Wrap in `Arc` to share across threads.
pub struct NdjsonSink {
    writer: Mutex<BufWriter<File>>,
}

impl NdjsonSink {
    /// Open or create a file at `path`. Each write is line-flushed so a
    /// long-running session survives a crash.
    pub fn open(path: impl AsRef<Path>) -> CoreResult<Self> {
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
            .map_err(CoreError::Io)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(f)),
        })
    }

    /// Convert a hub event into a recorded event and write one NDJSON
    /// line. Errors propagate as `CoreError::Backend`.
    pub fn write(&self, ev: &HubEvent, recorded_at: u64) -> CoreResult<()> {
        let recorded = hub_to_recorded(ev, recorded_at);
        let mut s = serde_json::to_string(&recorded)
            .map_err(|e| CoreError::Backend(format!("ndjson serialize: {e}")))?;
        s.push('\n');
        let mut w = self.writer.lock().expect("sink lock poisoned");
        w.write_all(s.as_bytes()).map_err(CoreError::Io)?;
        w.flush().map_err(CoreError::Io)?;
        Ok(())
    }
}

/// Build a subscriber closure that writes every event to the sink.
pub fn sink_subscriber(
    sink: std::sync::Arc<NdjsonSink>,
    clock: std::sync::Arc<dyn Fn() -> u64 + Send + Sync>,
) -> impl Fn(&HubEvent) + Send + Sync + 'static {
    move |event: &HubEvent| {
        let now = clock();
        if let Err(e) = sink.write(event, now) {
            eprintln!("ndjson sink: {e}");
        }
    }
}

fn hub_to_recorded(ev: &HubEvent, recorded_at: u64) -> RecordedEvent {
    match ev {
        HubEvent::Topology { tree } => RecordedEvent::Topology {
            tree: tree.clone(),
            recorded_at,
        },
        HubEvent::Observation { path: _, view } => RecordedEvent::Observation {
            view: view.clone(),
            recorded_at,
        },
        HubEvent::Edit {
            path,
            op,
            before,
            after,
        } => RecordedEvent::Edit {
            path: path.clone(),
            op: op.clone(),
            before: before.clone(),
            after: after.clone(),
            recorded_at,
        },
    }
}
