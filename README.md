# sememe

sememe is a debug harness for text-encoder models. It sits inside a forward
pass and lets you watch what the model is doing: it observes the intermediate
tensors — embeddings, attention activations, layer outputs — and, when you
need to, edits them surgically to test a hypothesis about where behaviour
comes from.

It is a working environment for experiments, not a product. The API is small
and will change as M1 is built out; see the [changelog](./CHANGELOG.md).

## Status

M4 is in. The workspace builds, the library exposes the core types, the
`Backend` seam, a `Telemetry` live hub with subscriber callbacks, and a
PyO3 bridge that loads a model, walks `named_modules`, and now installs
PyTorch forward hooks that stream `TensorView`s back across the boundary
as Arrow IPC. A ratatui TUI renders the live hub against any backend.
`sememe tui` launches it against `Qwen/Qwen3.5-0.8B`. M5 adds
`Backend::edit` and the CLI command that drives a live edit.

## Build

```sh
# full gate: format check, clippy, tests, Python wheel build + smoke
just gate

# or just build the workspace
cargo build --workspace
```

## Run

```sh
cargo run -p sememe-cli
# → sememe 0.0.0

# launch the live TUI against the demo Qwen3.5-0.8B
just tui
```

## Python setup (M2+)

The bridge needs a venv with `torch`, `transformers`, `pyo3`-compatible
Python, and `maturin`. `.venv/` is gitignored.

```sh
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install transformers pyarrow maturin
just python-build   # builds the wheel into the venv
```

## What is implemented

- `crates/sememe` — the library crate, with `sememe::version()`, the
  observable types (`ModulePath`, `TensorView`, `EditOp`, …), the
  `Backend` trait, an in-memory `Telemetry` hub with subscriber
  callbacks, and a generic `Harness<B>` that wires the two together.
- `crates/sememe-bridge` — PyO3 bridge that loads a real PyTorch
  model via `transformers.AutoModel`, walks `named_modules`, installs
  PyTorch forward hooks on every named module, runs one forward, and
  streams `TensorView`s back across the boundary as Arrow IPC.
  `edit` is stubbed until M5.
- `crates/sememe-tui` — ratatui TUI with three panes (tree | selected
  module's tensor stats | recent HubEvents), vim keys, NDJSON
  session-log sink. Subscribes to the harness's telemetry and
  repaints on every event.
- `apps/sememe-cli` — `sememe --version` and `sememe tui`.

## What is not (yet)

- Edit commands in the CLI (M5).
- Replay of recorded sessions (M6).
