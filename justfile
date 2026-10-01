# `just` runner — https://github.com/casey/just
# Run `just` to list available recipes.

set shell := ["zsh", "-cu"]

default:
    @just --list

# Format, lint, and test. The CI gate.
gate:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo test --workspace --all-features

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
