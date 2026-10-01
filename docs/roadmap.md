# Roadmap

> Living document: what the harness observes and edits, and how the pieces ship. No dates.

## Status

Early scaffold. The workspace builds; `crates/sememe` exposes `sememe::version()` and a
placeholder `Harness`, and `apps/sememe-cli` prints it. There is **no PyTorch bridge yet** —
no `Backend` seam and no implementation exist; nothing is observed or edited. M1 defines
the seam and the core types. M2 wires the first real implementation.

## Non-goals

- A training / fine-tuning tool. Sememe runs forward passes and edits tensors; it does
  not optimize parameters.
- A serving / inference runtime. It runs one forward at a time, by request, for
  diagnosis — not for throughput.
- A model-format converter. It attaches to a PyTorch model via a shim and stays in that
  representation; onboarding ONNX is a backend swap, not a conversion.
- A general ML framework. One seam, one purpose: watch and patch a text-encoder's
  intermediate tensors.

## Record and replay

Record and replay is the load-bearing feature that lets a user trust the harness on a
real model. If every forward pass, every hook fire, and every edit can be logged, the
same session replays into the TUI with the model off — exactly as it ran. The
decomposition lands one piece per milestone; **M1 through M5 record events but cannot yet
replay them. M6 is the first milestone that can replay.** Each milestone writes the
events it can into the same log; M6 reads them back.

The log is NDJSON. Each line is a `RecordedEvent` (serde-tagged): model topology, forward
passes, per-hook tensor observations, and edits at a module path. A `Recorder` writes it;
a `Replayer` reads it. Both run on the same `Backend` seam — M6's trick is a `Backend` that
replays from the log instead of running PyTorch, so the TUI and `Harness` are unchanged.
The in-memory telemetry store from M1 stays as the fast path; M4's NDJSON writer mirrors
it. Recording is intentionally cheaper than replay — no milestone gates on replay until
M6.

| Piece | Introduced | Notes |
| --- | --- | --- |
| Serializable record schema (serde-tagged `RecordedEvent`) | M1 | Pure Rust; the whole record format is this enum. |
| Topology record (`ModuleTree`, once, at attach) | M2 | Recorded the first time a backend exposes its module tree. |
| Observation records (per forward pass, per hook) | M3 | Streamed tensor records, written to the log as they fire. |
| Persisted session-log sink (NDJSON writer) | M4 | Recording leaves the in-memory store and lands on disk. |
| Edit records (path + `EditOp`, before / after) | M5 | Every applied edit becomes a log line. |
| Session-log reader and `Replayer` | M6 | Full record **and** replay; the replay `Backend`. |

## M1 — Core types in pure Rust

**Goal.** A user or a test can describe a module path, observe a tensor's stats, and
record an edit entirely in pure Rust, with no Python or UI in the loop.

**Crate-level changes.** `crates/sememe` leaves its placeholder and becomes the real core:
the leaf types, the `Backend` seam as a trait, an in-memory telemetry store, an
`EditLog`, and the serializable record schema. A stub in-memory `Backend` backs the
store in tests. Workspace deps add `serde`, `serde_json`. No new crate yet; the
placeholder `Harness` becomes generic over the `Backend` trait.

**Public API surface (sketch).**

```rust
pub struct ModulePath(String);            // "encoder.layer.3.attention"

pub struct TensorView {                   // an opaque stats record, never the tensor
    pub path: ModulePath,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
    pub min: f64, pub max: f64, pub mean: f64,
    pub samples: Vec<f64>,
}

pub enum EditOp { Zero, Scale(f64), Add(f64), Patch(TensorView) }

pub struct EditLog;                       // append-only edits, grouped by path
pub struct Telemetry;                     // in-memory tree + per-forward stats
pub struct Harness<B: Backend>;           // orchestrates any Backend
```

**Out of scope.** No PyO3, no model attached, no TUI, no replay. `TensorView` is
deliberately a stats record (min / max / mean / samples), not the raw tensor — the raw
tensor lives on the Python side. Observing and patching stats is enough to reason about
behaviour and cheap enough to record. The stub backend exists so `Harness` is
instantiable, so it is not the shipped PyTorch backend.

**Gate.** The workspace `gate` (`just gate` — `cargo fmt --all -- --check`, `cargo clippy
--workspace --all-targets --all-features -- -D warnings`, `cargo test --workspace
--all-features`) passes clean. No `Backend` exists today; M1 adds the trait and a stub
implementation so `Harness` is instantiable in tests.

## M2 — Bridge metadata only

**Goal.** The FFI boundary works end to end: load a real Python PyTorch model and return
its `named_modules()` tree as a Rust `ModuleTree`.

**Crate-level changes.** Adds `crates/sememe-bridge`, the PyO3 crate that owns the
boundary: it calls into Python, wraps a loaded model, and returns `named_modules()` as a
`ModuleTree`. The Python shim lives separately under `bindings/python` and is packaged as
a wheel via maturin. `crates/sememe` gains `Backend` implemented by the bridge (metadata
only). Workspace dep adds `pyo3`. The new crate is added to the root `Cargo.toml`
`members = [...]` so the workspace gate compiles and tests it. No UI yet.

**Public API surface (sketch).**

```rust
// crates/sememe-bridge, wraps PyO3:
pub fn load(path: &str) -> Result<PyModel>;     // attach to a real model
pub fn named_modules(model: &PyModel) -> Result<ModuleTree>;

// crates/sememe core:
pub struct ModuleTree { pub root: String, pub children: Vec<ModulePath> }
impl Backend for crate::bridge::PyBackend {
    fn named_modules(&self) -> Result<ModuleTree>;  // records topology once
}
```

**Out of scope.** No hooks, no tensor data, no edits — the bridge returns the *tree
structure*, not any tensor. M2 proves the boundary by shipping topology only; the
"model in, tree out" demo observes nothing inside a forward pass. The topology record is
written to the session log the first time a backend exposes its tree, but nothing
replays it yet.

**Gate.** The workspace `gate` (`just gate` — `cargo fmt --all -- --check`, `cargo clippy
--workspace --all-targets --all-features -- -D warnings`, `cargo test --workspace
--all-features`) passes clean. The new crate is in `members`; the gate actually exercises it.

## M3 — Bridge streams tensor records

**Goal.** A user can hook one module on a real model and watch a single forward pass
emit tensor records end to end.

**Crate-level changes.** `crates/sememe-bridge` now installs PyTorch forward hooks and
streams `TensorView` records over a Rust channel on every hook fire. Arrow IPC carries
the records across the PyO3 boundary (`arrow`, `arrow-pyarrow`). `crates/sememe` extends
`Backend` with a forward method returning that channel and records each `TensorView` to
the session log. No UI yet.

**Public API surface (sketch)**.

```rust
pub struct ModelInput {
    pub ids: Vec<i64>,
    pub type_ids: Option<Vec<i64>>,
}

impl Backend for crate::bridge::PyBackend {
    fn named_modules(&self) -> Result<ModuleTree>;
    fn run_forward(&self, input: &ModelInput) -> Result<Vec<TensorView>>;
}
```

**Out of scope.** The stream is one direction: record and observe. No edits yet, no TUI
to view the stream live, no replay. M3 records every hook fire into the session log but
cannot persist to disk nor play anything back — recording lands in memory only. The tiny
BERT-base used for the demo is a fixture for the demo, not a shipped fixture library.

**Gate.** The workspace `gate` passes clean, plus a check that the channel emits
records for a hooked layer.

## M4 — TUI renders the live model

**Goal.** A human can watch the live model: navigate the module tree, read a selected
module's tensor stats, and watch the hook and edit logs update.

**Crate-level changes.** Adds `crates/sememe-tui` (ratatui): module tree on the left,
selected module's tensor stats (shape, dtype, min / max / mean, sample values) in the
middle, hook and edit log on the right, vim keys to navigate — rendering whatever a
`Backend` streams. The new crate is added to the root `Cargo.toml` `members = [...]` so
the workspace gate compiles and tests it. M4 also adds the persisted NDJSON session-log
sink: recording leaves the in-memory store and lands on disk.

**Public API surface (sketch).**

```rust
// crates/sememe-tui
pub fn run(runtime, harness: &mut Harness<PyBackend>) -> Result<()>;
//   left: tree  |  middle: TensorView stats  |  right: EditLog / hook stream

// Recording now persists:
pub fn open_log(path: &Path) -> Recorder;   // NDJSON session writer
```

**Out of scope.** No edits are rendered or applied — the log is read-only on screen, and
`EditOp` has not yet touched any backend. The TUI reads a live backend; it cannot yet
replay a saved session (that is M6). M4 persists the session log to NDJSON, so every
forward and hook fire is now recorded to disk — but the file is write-only this
milestone.

**Gate.** The workspace `gate` passes clean across all three crates; both new crates are
registered in `members` before the gate runs.

## M5 — Surgical edits end to end

**Goal.** A user applies an `EditOp` at a named module path and watches the model's
output change exactly as predicted.

**Crate-level changes.** `crates/sememe` and `crates/sememe-bridge` add the mutation
half: `Backend::edit` applies an `EditOp` at a `ModulePath` through the Python model, and
the edit is recorded to the session log (path + op, with before / after stats).
`crates/sememe-cli` gains the command that drives a live edit. M5 closes the loop the
harness exists for.

**Public API surface (sketch)**.

```rust
impl Backend for crate::bridge::PyBackend {
    fn edit(&mut self, path: &ModulePath, op: &EditOp) -> Result<()>;
}
// Records the edit to the session log with before / after TensorViews.
```

**Out of scope.** The edit is applied in the live backend only — no editable TUI
controls, and no editable backend in a replay. The verification is external: run a
known-broken forward pass, apply an edit, watch the output change as predicted — a
scripted check, not a general edit UI. One edit per path per forward for now; several
simultaneous edits at one path are out.

**Gate.** The workspace `gate` passes clean across all three crates, plus an end-to-end
check that a known-broken forward changes as predicted after an edit.

## M6 — Record / replay

**Goal.** A full session can be saved to NDJSON and replayed into the TUI with the model
off, exactly as it ran.

**Crate-level changes.** `crates/sememe-bridge` gains a replay `Backend` — the same
`Backend` trait, backed by the NDJSON session log instead of PyTorch — so `Harness` and
the TUI are unchanged. `crates/sememe` adds the session-log reader and `Replayer`; the
recorded events form a deterministic replay stream. No new crate: this wires the final
implementation of the seam introduced in M1 and fed in M2 through M5.

**Public API surface (sketch)**.

```rust
pub struct Replayer;                        // reads NDJSON, emits RecordedEvent
pub struct ReplayBackend;                   // Backend backed by the replay stream
impl Backend for ReplayBackend {
    fn named_modules(&self) -> Result<ModuleTree>;    // from the topology record
    fn run_forward(&self, input) -> Result<Vec<TensorView>>;  // from saved forwards
    fn edit(&mut self, path, op) -> Result<()>;       // from saved edits
}
```

**Out of scope.** Replay is exact, not editable live — it shows what ran, not a new run.
It replays what was recorded; it does not record anything new. M6 is the first milestone
that replays, and only then, after M1 through M5 have been recording every forward pass,
hook fire, and edit.

**Gate.** The workspace `gate` passes clean across every member, plus a round-trip: save
a session and replay it back through `ReplayBackend` with the model off.
