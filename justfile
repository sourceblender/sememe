# Sememe's Python cockpit is the active development path. The Rust crates stay
# in the repository, with their gate available separately while the UI takes shape.
set shell := ["zsh", "-cu"]

default:
    @just --list

# Run the cockpit on a tiny built-in model; torch is not required.
demo:
    uv run sememe --mock

# Inspect a local model directory or a Hugging Face model id.
run model:
    uv run --extra torch sememe {{ model }}

# Fast Python checks, including the mock-only installation.
test:
    uv run pytest -q

# Exercise the real engine when the optional model stack is installed.
test-model:
    uv run --extra torch pytest -q tests/test_torch_engine.py

gate:
    uv run python -m compileall -q src/sememe
    just test
    uv build

# The earlier Rust prototype is preserved, without making it the cockpit gate.
rust-gate:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo test --workspace --all-features
