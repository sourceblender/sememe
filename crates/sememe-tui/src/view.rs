//! Pure-rust render model. The `render` function takes a snapshot of
//! the telemetry tree plus the `TuiState` and produces a `RenderModel`
//! — a description of what each pane should show. Tests exercise this
//! without needing a terminal; ratatui just turns the model into pixels.

use sememe::TensorView;
use sememe::telemetry::ModuleNode;

use crate::state::{TuiEvent, TuiState};

/// What each pane should show. The TUI runtime walks this and turns it
/// into ratatui widgets.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderModel {
    pub tree: Vec<TreeRow>,
    pub middle: MiddlePane,
    pub events: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TreeRow {
    pub depth: usize,
    pub label: String,
    pub selected: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MiddlePane {
    Empty,
    Node(MiddleNode),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MiddleNode {
    pub path: String,
    pub latest: Option<RenderedView>,
    pub history_len: usize,
    pub edits_len: usize,
    pub mean_delta: Option<f64>,
    pub forwards_since_last_obs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedView {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub samples: Vec<f64>,
}

impl From<&TensorView> for RenderedView {
    fn from(v: &TensorView) -> Self {
        Self {
            shape: v.shape.clone(),
            dtype: v.dtype.name().to_string(),
            min: v.min,
            max: v.max,
            mean: v.mean,
            samples: v.samples.clone(),
        }
    }
}

/// Build the render model from a snapshot of every telemetry tree node
/// (pre-order) and the current TUI state. `forward_count` is used for
/// "forwards since last observation" computation.
pub fn render(nodes: &[&ModuleNode], state: &TuiState, forward_count: u64) -> RenderModel {
    let tree: Vec<TreeRow> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| TreeRow {
            depth: depth_of(node.path().as_str()),
            label: node.name().to_string(),
            selected: i == state.selected,
        })
        .collect();

    let middle = match nodes.get(state.selected) {
        None => MiddlePane::Empty,
        Some(node) => MiddlePane::Node(MiddleNode {
            path: node.path().to_string(),
            latest: node.latest.as_ref().map(RenderedView::from),
            history_len: node.history.len(),
            edits_len: node.edits.len(),
            mean_delta: node.last_mean_delta(),
            forwards_since_last_obs: node.forwards_since_last_obs(forward_count),
        }),
    };

    let events: Vec<String> = state
        .events
        .iter()
        .map(|TuiEvent { line }| line.clone())
        .collect();

    RenderModel {
        tree,
        middle,
        events,
    }
}

fn depth_of(path: &str) -> usize {
    if path.is_empty() {
        0
    } else {
        path.split('.').filter(|s| !s.is_empty()).count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pane {
    Tree,
    Middle,
    Events,
}
