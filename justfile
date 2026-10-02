# `just` runner — https://github.com/casey/just
# Run `just` to list available recipes.

set shell := ["zsh", "-cu"]

# The model the bridge tests load. Override with SEMEME_TEST_MODEL=<snapshot dir>;
# by default it is the cached Qwen3.5-0.8B snapshot, resolved offline from the
# Hugging Face cache, so no machine-specific path lives in the repo.
test_model_id := "Qwen/Qwen3.5-0.8B"

default:
    @just --list

# Format, lint, and test. The CI gate. Builds the maturin wheel into
# the venv before cargo test so PyO3 can resolve a Python interpreter.
gate:
    just python-build
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    just test
    just python-smoke

# The workspace tests, with the embedded interpreter pointed at the project venv
# and the bridge tests pointed at a real model. Missing either is a failure.
test:
    #!/usr/bin/env zsh
    set -euo pipefail
    model="$(just test-model)"
    VIRTUAL_ENV="$(pwd)/.venv" SEMEME_TEST_MODEL="$model" PYO3_PYTHON="$(pwd)/.venv/bin/python" cargo test --workspace --all-features

# Print the test model's snapshot directory (SEMEME_TEST_MODEL if set). Fails,
# rather than printing nothing, when the model cannot be found.
test-model:
    #!/usr/bin/env zsh
    set -euo pipefail
    if [[ -n "${SEMEME_TEST_MODEL:-}" ]]; then
      dir="$SEMEME_TEST_MODEL"
    else
      dir="$(.venv/bin/python -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], local_files_only=True))' {{ test_model_id }})"
    fi
    if [[ ! -d "$dir" ]]; then
      print -u2 "test model directory not found: ${dir:-<empty>}"
      exit 1
    fi
    print -r -- "$dir"

# Build the maturin wheel into the active venv. Required before
# `cargo test -p sememe-bridge` because the integration test loads the
# Python extension at runtime.
python-build:
    .venv/bin/maturin develop

# Smoke test that the wheel loads and `named_modules` returns a tree.
python-smoke:
    .venv/bin/python -c "import sys, sememe_bridge as sb; m = sb.load(sys.argv[1]); names = m.named_modules(); assert len(names) > 0; print('full OK; modules:', len(names))" "$(just test-model)"

# Run all Python-side gates: build the wheel, then a smoke load. CI runs
# this after `cargo test --workspace` so the gate is self-contained.
python-gate: python-build python-smoke

# Build everything.
build:
    cargo build --workspace

# Run the CLI.
run *ARGS:
    cargo run -p sememe-cli -- {{ ARGS }}

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
