# sememe

sememe is a debug harness for text-encoder models. It sits inside a forward
pass and lets you watch what the model is doing: it observes the intermediate
tensors — embeddings, attention activations, layer outputs — and, when you
need to, edits them surgically to test a hypothesis about where behaviour
comes from.

It is a working environment for experiments, not a product. The API is small
and will change as M1 is built out; see the [changelog](./CHANGELOG.md).

## Status

M1 is in. The workspace builds, the library exposes the core types, the
`Backend` seam, and a `Telemetry` live hub with subscriber callbacks. Tests
exercise the surface end to end through a stub `Backend`. There is **no
PyTorch bridge yet** — sememe has no connection to a running model, so
nothing is observed from a real network until M2.

## Build

```sh
# full gate: format check, clippy, tests
just gate

# or just build the workspace
cargo build --workspace
```

## Run

```sh
cargo run -p sememe-cli
# → sememe 0.0.0
```

## What is implemented

- `crates/sememe` — the library crate, with `sememe::version()`, the
  observable types (`ModulePath`, `TensorView`, `EditOp`, …), the
  `Backend` trait, an in-memory `Telemetry` hub with subscriber
  callbacks, and a generic `Harness<B>` that wires the two together.
  See [the roadmap](./docs/roadmap.md) for M1's full surface.
- `apps/sememe-cli` — the `sememe` binary, which prints the version.

## What is not (yet)

- A PyTorch bridge to attach to an actual model.
- Hooks that stream observations from a live forward pass.
- A TUI that renders the hub live.
- Commands in the CLI beyond printing the version.
