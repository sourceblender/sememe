# `just` runner — https://github.com/casey/just
# Run `just` to list available recipes.

set shell := ["zsh", "-cu"]

default:
    @just --list

# Format, lint, and test. The CI gate. Builds the maturin wheel into
# the venv before cargo test so PyO3 can resolve a Python interpreter.
gate:
    just python-build
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    PYO3_PYTHON="$(pwd)/.venv/bin/python" cargo test --workspace --all-features
    just python-smoke

# Build the maturin wheel into the active venv. Required before
# `cargo test -p sememe-bridge` because the integration test loads the
# Python extension at runtime.
python-build:
    .venv/bin/maturin develop

# Smoke test that the wheel loads, walks the tree, and runs one forward
# pass with hooks firing through Arrow IPC.
python-smoke:
    .venv/bin/python -c "import sememe_bridge as sb; m = sb.load(sb.demo_model_path()); names = m.named_modules(); assert len(names) > 0; data = sb.run_forward_with_hooks(m, [101, 2024, 3056], names); assert len(data) > 1000; print('full OK; modules:', len(names), 'bytes:', len(data))"

# Run all Python-side gates: build the wheel, then a smoke load. CI runs
# this after `cargo test --workspace` so the gate is self-contained.
python-gate: python-build python-smoke

# Build everything.
build:
    cargo build --workspace

# Run the CLI.
run *ARGS:
    cargo run -p sememe-cli -- {{ ARGS }}

# Launch the live TUI against the demo Qwen model. Requires the venv
# and `just python-build` to have run.
tui:
    PYO3_PYTHON="$(pwd)/.venv/bin/python" cargo run -p sememe-cli -- tui

# Apply formatting.
fmt:
    cargo fmt --all

# Auto-fix where possible, then gate.
fix:
    cargo fmt --all
    cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged

# Run the Criterion benchmark suite (extra args go to Criterion).
bench *ARGS:
    cargo bench -p sememe -- {{ ARGS }}

# Line coverage summary (needs cargo-llvm-cov).
coverage:
    cargo llvm-cov --workspace --all-features --summary-only
