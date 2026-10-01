# sememe

sememe is a debug harness for text-encoder models. It sits inside a forward
pass and lets you watch what the model is doing: it observes the intermediate
tensors — embeddings, attention activations, layer outputs — and, when you
need to, edits them surgically to test a hypothesis about where behaviour
comes from.

It is a working environment for experiments, not a product. The API is small
and will change as M1 is built out; see the [changelog](./CHANGELOG.md).

## Status

Early scaffold. The workspace builds, the library exposes a version accessor,
and the CLI prints it. There is **no PyTorch bridge yet** — sememe has no
connection to a running model, so nothing is observed or edited until M1.

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

- `crates/sememe` — the library crate, with `sememe::version()` and a
  placeholder `Harness` handle.
- `apps/sememe-cli` — the `sememe` binary, which prints the version.

## What is not (yet)

- A PyTorch bridge to attach to an actual model.
- Any way to read or write intermediate tensors.
- Commands in the CLI beyond printing the version.

The shape of the observed-and-edited tensor API — its surface, what a
"patch" looks like, how errors surface through `thiserror` — is decided in
M1.
