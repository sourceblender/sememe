# Roadmap

> Living document: what the harness observes and edits, and how the pieces ship. No dates.

## Status

M1 and M2 are built. The core types, the live `Telemetry` hub and the record schema are
pure Rust and tested. The PyO3 bridge loads a real PyTorch model and walks its
`named_modules()`. M2.1 hardens that bridge before M3 starts. The first model is
Qwen3.5-0.8B: same family and module layout as the larger Qwen3.5 models, small enough to
test in seconds.

## What sememe is

A debugger for a model's forward pass, the way gdb is for a program. It is not an
inference runtime and not a training tool. It exposes PyTorch's hook points so you can
see what passes through any module, and change it to test a hypothesis about where a
behaviour comes from. The goal is investigation that today's tooling doesn't reach: most
LoRA and fine-tuning work happens downstream, but nothing says a change can't act inside
the text encoder or any other module.

Three verbs:

- **Observe.** Hooks at any module path record what passes through: per-layer,
  per-token stats, and full tensors when asked.
- **Compare.** The same prompt through two models, or two prompts through one, diffed
  module by module, to find where a fine-tune or a LoRA actually acts.
- **Intervene.** Zero, scale or replace a module's output, swap a block from another
  checkpoint, or add a delta at any module, then watch what changes downstream.

Three ways in, all on the same hook points:

- **Breakpoints.** `break <module path>` pauses the forward pass there. You inspect the
  tensor, optionally edit it, then `step` to the next module or `continue`.
- **Sidecars.** An external process binds to a hook point over a local socket, in any
  language. A *watch* sidecar sees each event and never holds up the run. An *intercept*
  sidecar pauses the run and must answer with a replacement or a pass-through before a
  deadline. A LoRA delta on the text encoder, a probe, or a logger is just a sidecar.
- **The TUI.** One client of the harness among others: module tree, tensor stats,
  forward-pass timeline, breakpoints and sidecars.

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

Recording is cheap. The recorder is a built-in watch sidecar that writes every event to
an NDJSON session log: model topology, forward passes, per-hook tensor observations, and
every intervention with which breakpoint or sidecar made it and what changed.

Replay is not free, and it gets its own milestone and proof. It needs a defined event
order, tensor payloads that can be read back, the intervention decisions as recorded,
and a model-off `Backend` that reproduces exactly the state the TUI showed. M1 through M5
record what they can into the same log. M6 is the first milestone that replays.

| Piece | Introduced | Notes |
| --- | --- | --- |
| Serializable record schema (serde-tagged `RecordedEvent`) | M1 | Pure Rust; the whole record format is this enum. |
| Topology record (`ModuleTree`, once, at attach) | M2 | Recorded the first time a backend exposes its module tree. |
| Observation records (per forward pass, per hook) | M3 | Written as hooks fire. |
| Persisted session-log sink (NDJSON writer) | M4 | Recording leaves the in-memory store and lands on disk. |
| Intervention records (who acted, path, before / after) | M5 | Breakpoint edits and intercept sidecars alike. |
| Session-log reader, `Replayer` and replay `Backend` | M6 | Replay, with its own proof gate. |

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

## M2.1 — The bridge proves what it claims

**Goal.** The bridge's tests load the real model every time they run, and fail when they
can't.

**What changes.**

- A Rust-embedded interpreter adds `VIRTUAL_ENV`'s site-packages before importing
  `transformers`. Before this, `cargo test` started the base Python, failed to import
  `transformers`, printed "skipping" and passed. The Rust `PyBackend` had never loaded
  the model under `just gate`.
- The smoke tests never skip. A missing model or missing Python dependency fails them.
- `named_modules` uses Python's iterator protocol, so only `StopIteration` ends the walk.
  The old loop read any error as the end of the tree and returned a partial one.
- `ModuleTree.root` is the model's class name, not the first child's first segment.
- No machine-specific path in the repo. The test model comes from `SEMEME_TEST_MODEL`,
  or `just test` resolves the cached Qwen3.5-0.8B snapshot offline. The same tests can
  point at a larger Qwen model later.

**Gate.** `just gate`, with the bridge tests loading Qwen3.5-0.8B. Each fix is
red-proofed: putting the old behaviour back fails a test.

## M3 — Hooks, breakpoints, step and continue

**Goal.** On the real model, a user can observe any module's output during one forward
pass, pause there, and change it, and see the change downstream.

**Crate-level changes.** The bridge installs PyTorch forward hooks at requested module
paths and streams observations into `Telemetry`. Hook lifetime is owned by the bridge: a
forward that fails, or is cancelled, still removes every hook it installed. `Harness`
gains breakpoints, `step` and `continue`, driven in-process from Rust (the TUI comes in
M4). Each event carries the module path, a forward-pass ID and a typed tensor
description.

**First proof.** One prompt through Qwen3.5-0.8B's `language_model`: record every
layer's output, then zero (or scale) one whole module's output at a breakpoint and show
which downstream modules change. That proves observe, break and intervene end to end on
a module boundary `named_modules` actually exposes.

**Unverified until a live tensor read.** A single attention head is not a module
boundary once heads are combined. The expected tap is a pre-forward hook on the attention
output projection, reshaped to `[.., n_heads, head_dim]` so one head can be sliced. It
stays out of the M3 gate until Qwen3.5's actual attention shapes are read.

**Gate.** `just gate`, plus the first proof against the real model, and a test that a
forward which raises leaves no hooks attached.

## M4 — The TUI

**Goal.** A human drives the debugger: browse the module tree, set breakpoints, read and
edit the tensor at a breakpoint, and follow the forward pass.

**Crate-level changes.** Adds `crates/sememe-tui` (ratatui): module tree, tensor stats
(shape, dtype, min / max / mean, samples), forward-pass timeline, and the breakpoint and
sidecar lists, with vim keys. The TUI is a client of `Harness`, not a special case. M4
also adds the NDJSON session-log sink, so recording lands on disk.

**Gate.** `just gate` across all crates, plus a scripted TUI session against the real
model that sets a breakpoint, edits, and continues.

## M5 — The sidecar protocol

**Goal.** An external process, in any language, can watch or intercept any hook point.

**The contract.** A local socket. JSON control messages; tensors as Arrow or shared
memory so they aren't copied. Each event carries the module path, the forward-pass ID and
a typed tensor description.

- **watch** is asynchronous. The sidecar receives the event; the run never waits.
- **intercept** is synchronous. The run pauses until the sidecar answers with a
  replacement tensor or a pass-through, or until its deadline passes. A missed deadline
  is a recorded failure, never a silent pass-through.

Every intercept is recorded: which sidecar acted, at what path, and what changed.
Breakpoint edits from M3 and M4 go through the same intervention record.

**Gate.** `just gate`, plus two reference sidecars against the real model: a watcher
that logs every event, and an interceptor that applies a known edit and is shown to
change the output.

## M6 — Replay

**Goal.** A recorded session replays into the TUI with the model off, exactly as it ran.

**Crate-level changes.** `crates/sememe` adds the session-log reader and `Replayer`. A
replay `Backend`, the same trait backed by the log instead of PyTorch, feeds `Harness`
and the TUI unchanged.

**What it requires.** A defined event order; tensor payloads that can be read back;
the intervention decisions as recorded; and a backend that reproduces the TUI's state
from the log alone.

**Gate.** Its own proof, beyond `just gate`: record a session that includes breakpoints,
edits and an intercept sidecar, replay it with the model off, and show the replayed TUI
state equals the live one, event by event.
