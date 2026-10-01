use serde::{Deserialize, Serialize};

use crate::edit::EditOp;
use crate::input::ModelInput;
use crate::path::ModulePath;
use crate::topology::ModuleTree;
use crate::view::TensorView;

/// Every event written to the NDJSON session log (M4+) and read back by
/// the replay backend (M6). The schema is locked in M1 so M4's writer and
/// M6's reader don't have to negotiate the on-disk format later.
///
/// M1 emits none of these — the field shapes are real, just unused. M2
/// emits `Topology`, M3 emits `Forward` and `Observation`, M5 emits `Edit`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum RecordedEvent {
    Topology {
        tree: ModuleTree,
        recorded_at: u64,
    },
    Forward {
        input: ModelInput,
        views: Vec<TensorView>,
        recorded_at: u64,
    },
    Observation {
        view: TensorView,
        recorded_at: u64,
    },
    Edit {
        path: ModulePath,
        op: EditOp,
        before: Option<TensorView>,
        after: Box<Option<TensorView>>,
        recorded_at: u64,
    },
}
