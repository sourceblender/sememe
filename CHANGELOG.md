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
