# Roadmap

> Living document: what the harness observes and edits, and how the pieces ship. No dates.

## Status

M1 and M2 are built. The core types, the live `Telemetry` hub and the record schema are
pure Rust and tested. The PyO3 bridge loads a real PyTorch model and walks its
`named_modules()`. M2.1 hardens that bridge. From here sememe grows slice by slice, starting with the
TUI (see How we build from here). The first model is
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

Replay is not free, and it gets its own slice and proof. It needs a defined event order,
tensor payloads that can be read back, the intervention decisions as recorded, and a
model-off `Backend` that reproduces exactly the state the TUI showed. The record schema
from M1 and the topology record from M2 already exist; the rest arrives with the features
that produce it.

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

## How we build from here

PyTorch's hook surface is too large to design up front, so sememe grows from the app
outward. First a cockpit you can sit in front of, mocked where data doesn't exist yet;
then real data, one panel at a time, each slice chosen from a real question asked while
using it. A panel without real data says `MOCK` in its title, so nothing on screen
pretends to be measured.

## Slice 1 — The cockpit

**Goal.** Load Qwen3.5-0.8B and browse it as an architecture, with the whole layout in
place.

**Layout.**

- **Tabs**: Architecture, Tables, Decode. Hooks comes later.
- **Architecture**: the model as blocks — embedding, the decoder blocks (each split into
  attention and MLP), final norm and LM head, with the vision tower as a collapsed
  branch. Color marks component type. Every block maps to a real module path.
- **Sidebar**: the selected block's class, parameter shapes, dtypes and counts. Enter
  drills into a block (attention's projections, the MLP's gate / up / down). Mouse and
  keyboard select the same way.
- **Search**: `/` jumps to a module by path.
- **Monitoring strip**: a few lines at the bottom with running line charts for prefill
  and decode throughput, per-step latency and memory. Not a benchmark: it's how you see
  what a hook costs.

**Real in slice 1**: the block map from the actual module paths, and the sidebar's static
details. **Mocked**: Tables, Decode and the monitoring strip, until their slices land.

**Gate.** `just gate`, plus render tests that run without a terminal (the view builds a
plain render model; ratatui only draws it), and the real Qwen load.

## Backlog, in rough order

Each becomes a slice when we pick it, with its own gate.

- **Static model data across the bridge**: config, every parameter's shape, dtype and
  count, weight stats. Feeds the sidebar and the Tables tab.
- **Decode**: run a prompt, show tokens and the top-k next tokens.
- **Monitoring for real**: prefill and decode timing and memory into the strip.
- **Hooks at module paths**: observe a module's output during a forward pass. The bridge
  owns hook lifetime: a forward that fails or is cancelled still removes every hook.
- **Breakpoints**: pause at a module, inspect, edit, `step` or `continue`. The first
  intervention is a whole-module output (zero or scale), shown downstream.
- **Chart markers**: where a hook or edit acted, marked on the monitoring strip.
- **Compare**: one prompt through two models, or two prompts through one, diffed by module.
- **Sidecars**: external processes on a local socket. *watch* is asynchronous; *intercept*
  pauses the run until the sidecar returns a replacement or a pass-through, or its deadline
  passes (a missed deadline is a recorded failure). Each event carries the module path, a
  forward-pass ID and a typed tensor description; every intercept is recorded with who
  acted and what changed.
- **Record and replay**: the recorder sidecar, then replay with its own proof — record a
  session with breakpoints, edits and an intercept, replay it with the model off, and show
  the replayed TUI state equals the live one, event by event.
- **Per-head taps**: unverified. A head is not a module boundary once heads are combined;
  the expected tap is a pre-forward hook on the attention output projection reshaped to
  `[.., n_heads, head_dim]`. Not shown in the UI until Qwen3.5's real attention shapes are read.
