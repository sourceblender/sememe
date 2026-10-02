# Changelog

All notable changes to sememe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial workspace scaffold (`crates/sememe`, `apps/sememe-cli`).
- `sememe::version()` accessor and a smoke test.
- M1 core types: `ModulePath`, `TensorDType`, `TensorView`, `EditOp`,
  `ModuleTree`, `ModelInput`, `Error` (`thiserror`).
- `Backend` trait with the full future surface (`named_modules`,
  `run_forward`, `edit`); `StubBackend` for tests behind the `test-util`
  feature.
- `Telemetry` live hub: nested `ModuleNode` tree, bounded history ring
  (default cap 256), sync `Fn` subscriber callbacks for `HubEvent`
  broadcasts.
- `Harness<B: Backend>` orchestrating a backend and the hub.
- `RecordedEvent` serde-tagged enum covering topology, forward,
  observation, and edit events for the future NDJSON writer (M4) and
  replay backend (M6).
- Workspace deps `serde` and `serde_json`.
- M2 bridge: `crates/sememe-bridge` (PyO3, abi3-py312) exposing
  `PyBackend`. Loads via `transformers.AutoModel.from_pretrained` and
  implements `Backend::named_modules` against a real PyTorch model.
  `run_forward` and `edit` return `Error::Unsupported` until M3 and
  M5. maturin-driven wheel build into `.venv/`. Workspace dep `pyo3`.

### Fixed
- M2.1: the bridge tests load the real model and fail when they can't. A
  Rust-embedded interpreter now adds `VIRTUAL_ENV`'s site-packages, so
  `cargo test` no longer imports from the base Python, fails, prints
  "skipping" and passes.
- `named_modules` returns an error raised mid-walk instead of a partial tree.
- `ModuleTree.root` is the model's class name, not its first child's name.
- No machine-specific model path in the repo: `SEMEME_TEST_MODEL`, or `just
  test` resolves the cached Qwen3.5-0.8B snapshot offline. `demo_model_path`
  is removed from the bridge.

### Changed
- The roadmap reshapes M3–M6 around a debugger: breakpoints with step and
  continue (M3), the TUI as one client (M4), watch and intercept sidecars
  (M5), and replay with its own proof gate (M6).
- Python smoke test loads the demo `Qwen/Qwen3.5-0.8B` snapshot and
  asserts the module tree includes both `language_model` and `visual`
  branches.
