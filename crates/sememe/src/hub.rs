use serde::{Deserialize, Serialize};

use crate::edit::EditOp;
use crate::path::ModulePath;
use crate::topology::ModuleTree;
use crate::view::TensorView;

/// An event broadcast to live subscribers of `Telemetry`. Distinct from
/// [`crate::recorded::RecordedEvent`] — hub events are transient and in
/// memory; recorded events are the on-disk schema for persistence and M6
/// replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum HubEvent {
    Topology {
        tree: ModuleTree,
    },
    Observation {
        path: ModulePath,
        view: TensorView,
    },
    Edit {
        path: ModulePath,
        op: EditOp,
        before: Option<TensorView>,
        after: Box<Option<TensorView>>,
    },
}

/// Identifier returned by `Telemetry::subscribe`. Used to unsubscribe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubscriptionId(pub u64);
