use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use crate::edit::EditOp;
use crate::error::Result;
use crate::hub::{HubEvent, SubscriptionId};
use crate::path::ModulePath;
use crate::topology::ModuleTree;
use crate::view::TensorView;

/// Live nerve center. The M4 TUI subscribes to this; the M3 bridge writes
/// to this. Tree-shaped so the TUI's left pane is a direct traversal.
///
/// Per-node history is bounded (`history_cap`, default 256) so a long
/// debug session doesn't grow without limit. Edits are unbounded — they
/// are rare and small, and the append-only log is the point.
pub struct Telemetry {
    root: ModuleNode,
    current_forward: u64,
    history_cap: usize,
    next_sub_id: Mutex<u64>,
    subscribers: Mutex<Vec<(SubscriptionId, Subscriber)>>,
}

type Subscriber = Box<dyn Fn(&HubEvent) + Send + Sync + 'static>;

impl Telemetry {
    /// Create an empty hub. The root node exists so the tree is never
    /// empty; intermediate nodes are created on the first observation
    /// that names them.
    pub fn new() -> Self {
        Self::with_history_cap(256)
    }

    pub fn with_history_cap(history_cap: usize) -> Self {
        Self {
            root: ModuleNode::new("", ModulePath::new("")),
            current_forward: 0,
            history_cap,
            next_sub_id: Mutex::new(1),
            subscribers: Mutex::new(Vec::new()),
        }
    }

    pub fn current_forward(&self) -> u64 {
        self.current_forward
    }

    pub fn history_cap(&self) -> usize {
        self.history_cap
    }

    /// Bump the forward counter and return the new index. The harness
    /// calls this at the start of every `forward` and stamps the index
    /// on each `TensorView` before handing it to `record_observation`.
    pub fn begin_forward(&mut self) -> u64 {
        self.current_forward += 1;
        self.current_forward
    }

    /// Root node, always present.
    pub fn root(&self) -> &ModuleNode {
        &self.root
    }

    /// Look up a node by exact path. Returns `None` if no observation or
    /// topology record has touched this branch yet.
    pub fn node_at(&self, path: &ModulePath) -> Option<&ModuleNode> {
        let mut node = &self.root;
        for seg in path.as_str().split('.').filter(|s| !s.is_empty()) {
            node = node.children.get(seg)?;
        }
        Some(node)
    }

    /// Pre-order traversal of every node in the tree.
    pub fn iter(&self) -> impl Iterator<Item = &ModuleNode> {
        let mut out = Vec::new();
        self.root.walk_preorder(&mut out);
        out.into_iter()
    }

    /// Record a topology. Replaces the tree and notifies subscribers.
    /// Called by `Harness::topology` once `Backend::named_modules`
    /// succeeds.
    pub fn record_topology(&mut self, tree: &ModuleTree) -> Result<()> {
        self.root = ModuleNode::new("", ModulePath::new(""));
        for path in &tree.children {
            self.root.ensure_path(path.segments());
        }
        self.dispatch(&HubEvent::Topology { tree: tree.clone() });
        Ok(())
    }

    /// Record a per-hook observation. Walks the path, creating
    /// intermediate nodes if they don't exist yet. Replaces `latest`,
    /// pushes the previous `latest` onto the ring, drops the oldest past
    /// `history_cap`.
    pub fn record_observation(&mut self, mut view: TensorView) {
        let path = view.path.clone();
        let forward_index = self.current_forward;
        view.forward_index = forward_index;
        let node = self.root.ensure_path(path.segments());
        node.latest = Some(view.clone());
        if node.history.len() == self.history_cap {
            node.history.pop_front();
        }
        node.history.push_back(view.clone());
        self.dispatch(&HubEvent::Observation { path, view });
    }

    /// Record an edit. Captures the current `latest` as `before` and
    /// leaves `after` as `None` — the bridge re-reads and fills it in M5.
    pub fn record_edit(&mut self, path: &ModulePath, op: EditOp) {
        let before = self.root.ensure_path(path.segments()).latest.clone();
        self.root
            .ensure_path(path.segments())
            .edits
            .push(op.clone());
        self.dispatch(&HubEvent::Edit {
            path: path.clone(),
            op,
            before,
            after: Box::new(None),
        });
    }

    /// Register a callback invoked synchronously on every event. Returns
    /// the id used to unsubscribe. Callbacks run on the recorder's
    /// thread — keep them short.
    pub fn subscribe<F>(&self, callback: F) -> SubscriptionId
    where
        F: Fn(&HubEvent) + Send + Sync + 'static,
    {
        let mut next = self
            .next_sub_id
            .lock()
            .expect("subscription id lock poisoned");
        let id = SubscriptionId(*next);
        *next += 1;
        self.subscribers
            .lock()
            .expect("subscriber lock poisoned")
            .push((id, Box::new(callback)));
        id
    }

    pub fn unsubscribe(&self, id: SubscriptionId) {
        self.subscribers
            .lock()
            .expect("subscriber lock poisoned")
            .retain(|(sid, _)| *sid != id);
    }

    fn dispatch(&self, event: &HubEvent) {
        let snapshot = self.subscribers.lock().expect("subscriber lock poisoned");
        for (_, cb) in snapshot.iter() {
            cb(event);
        }
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

/// One node in the live tree. Public so callers (the TUI, tests) can read
/// `latest`, `history`, `edits`, and `children` directly.
#[derive(Debug)]
pub struct ModuleNode {
    name: String,
    path: ModulePath,
    pub latest: Option<TensorView>,
    pub history: VecDeque<TensorView>,
    pub edits: Vec<EditOp>,
    pub children: BTreeMap<String, ModuleNode>,
}

impl ModuleNode {
    fn new(name: impl Into<String>, path: ModulePath) -> Self {
        Self {
            name: name.into(),
            path,
            latest: None,
            history: VecDeque::new(),
            edits: Vec::new(),
            children: BTreeMap::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn path(&self) -> &ModulePath {
        &self.path
    }

    /// Difference between the most recent observation's mean and the
    /// previous one's. `None` if fewer than two observations have been
    /// recorded.
    pub fn last_mean_delta(&self) -> Option<f64> {
        let h = &self.history;
        if h.len() < 2 {
            return None;
        }
        let prev = &h[h.len() - 2];
        let curr = &h[h.len() - 1];
        Some(curr.mean - prev.mean)
    }

    /// Forwards since this node last saw an observation. `current` is
    /// `Telemetry::current_forward()` at the moment of the call.
    /// Returns `current` if the node has never been observed.
    pub fn forwards_since_last_obs(&self, current: u64) -> u64 {
        match &self.latest {
            None => current,
            Some(view) => current.saturating_sub(view.forward_index),
        }
    }

    fn ensure_path<'a>(
        &'a mut self,
        mut segments: impl Iterator<Item = &'a str>,
    ) -> &'a mut ModuleNode {
        let first = segments.next();
        match first {
            None => self,
            Some(seg) => {
                let base = self.path.as_str().trim_end_matches('.');
                let new_path = if base.is_empty() {
                    ModulePath::new(seg)
                } else {
                    ModulePath::new(format!("{base}.{seg}"))
                };
                let entry = self
                    .children
                    .entry(seg.to_string())
                    .or_insert_with(|| ModuleNode::new(seg, new_path));
                entry.ensure_path(segments)
            }
        }
    }

    fn walk_preorder<'a>(&'a self, out: &mut Vec<&'a ModuleNode>) {
        out.push(self);
        for child in self.children.values() {
            child.walk_preorder(out);
        }
    }
}
