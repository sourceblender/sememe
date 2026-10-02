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
- Python smoke test loads the demo `Qwen/Qwen3.5-0.8B` snapshot and
  asserts the module tree includes both `language_model` and `visual`
  branches.
- M3 hook firing: `run_forward_with_hooks` installs PyTorch forward
  hooks on every named module, runs one pass, and returns the captured
  `TensorView` stats as Arrow IPC bytes. Rust decodes via Arrow's
  `StreamReader`. `Backend::run_forward` is now live end to end against
  a real model. Hub `HubEvent::Observation` events fire as each view
  lands. Workspace dep `arrow` (ipc only, default-features = false).
- Bridge module renamed to `_native` to avoid the maturin/Python
  source name collision; the wheel's Python `__init__.py` re-exports
  it as `sememe_bridge`.
- M4 TUI: `crates/sememe-tui` (ratatui + crossterm). Three-pane
  layout (tree | selected module's tensor stats | recent HubEvents),
  vim keys (j/k, g/G, q), repaints on every `HubEvent` from the
  harness's telemetry. NDJSON session-log sink writes each event to
  disk as a serde-tagged `RecordedEvent` line. New `sememe tui`
  subcommand in `apps/sememe-cli` loads the demo Qwen3.5-0.8B and
  launches the TUI. Workspace deps `ratatui`, `crossterm`.
