use std::collections::VecDeque;

use sememe::HubEvent;

/// Mutable state held by the TUI runtime. The render layer is pure and
/// reads from this struct; the runtime layer (ratatui event loop)
/// mutates it.
pub struct TuiState {
    /// Index into the pre-order traversal of the telemetry tree.
    pub selected: usize,
    /// Buffered `HubEvent`s — most recent at the end. Capped.
    pub events: VecDeque<TuiEvent>,
    pub events_cap: usize,
    /// `true` if the user asked to quit.
    pub quit: bool,
    /// Forward-counter snapshot at the last repaint.
    pub forward_count: u64,
}

impl Default for TuiState {
    fn default() -> Self {
        Self::new()
    }
}

impl TuiState {
    pub fn new() -> Self {
        Self {
            selected: 0,
            events: VecDeque::new(),
            events_cap: 256,
            quit: false,
            forward_count: 0,
        }
    }

    pub fn move_selection(&mut self, delta: i64, node_count: usize) {
        if node_count == 0 {
            self.selected = 0;
            return;
        }
        let n = node_count as i64;
        let mut cur = self.selected as i64 + delta;
        if cur < 0 {
            cur = 0;
        }
        if cur >= n {
            cur = n - 1;
        }
        self.selected = cur as usize;
    }

    /// Push a new hub event into the right pane buffer. Drops the
    /// oldest past `events_cap`.
    pub fn push_event(&mut self, ev: HubEvent) {
        self.events.push_back(TuiEvent::from(ev));
        if self.events.len() > self.events_cap {
            self.events.pop_front();
        }
    }
}

/// One row in the right-pane event log. Compact one-line representation.
pub struct TuiEvent {
    pub line: String,
}

impl From<HubEvent> for TuiEvent {
    fn from(ev: HubEvent) -> Self {
        let line = match &ev {
            HubEvent::Topology { tree } => format!(
                "topology  {} modules  root={}",
                tree.children.len(),
                tree.root
            ),
            HubEvent::Observation { path, view } => format!(
                "obs       {:30}  shape={:?}  mean={:.4}  min={:.3}  max={:.3}",
                truncate(path.as_str(), 30),
                view.shape,
                view.mean,
                view.min,
                view.max,
            ),
            HubEvent::Edit {
                path,
                op,
                before,
                after,
            } => {
                let op = match op {
                    sememe::EditOp::Zero => "Zero".to_string(),
                    sememe::EditOp::Scale(k) => format!("Scale({k})"),
                    sememe::EditOp::Add(k) => format!("Add({k})"),
                    sememe::EditOp::Patch(_) => "Patch".to_string(),
                };
                let has_after = after.is_some();
                let has_before = before.is_some();
                format!(
                    "edit      {:30}  op={:<14}  before={}  after={}",
                    truncate(path.as_str(), 30),
                    op,
                    if has_before { "yes" } else { "no" },
                    if has_after { "yes" } else { "no" },
                )
            }
        };
        Self { line }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}
