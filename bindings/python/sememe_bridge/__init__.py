"""Public surface of the sememe-bridge wheel.

The compiled extension lives at `sememe_bridge._native`. This thin
wrapper re-exports it so callers see a normal Python package.
"""

from __future__ import annotations

from . import _native  # noqa: F401

__all__ = ["load", "named_modules", "run_forward_with_hooks", "demo_model_path"]


def load(path: str):
    """Load a HuggingFace transformers model from a snapshot directory."""
    return _native.load(path)


def named_modules(model) -> list[str]:
    """Depth-first list of every module's dotted name in `model`."""
    return _native.named_modules(model)


def run_forward_with_hooks(model, ids: list[int], paths: list[str]) -> bytes:
    """Run a forward pass with hooks on every named module. Returns Arrow IPC bytes."""
    return _native.run_forward_with_hooks(model, ids, paths)


def demo_model_path() -> str:
    """The Qwen3.5-0.8B snapshot path the integration tests use."""
    return _native.demo_model_path()
