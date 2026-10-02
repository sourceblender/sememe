"""PyTorch model inspection behind the UI's small Engine interface.

Importing this module does not import torch or transformers. The cockpit can
run in mock mode without the optional model dependencies installed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .api import EngineError, ModelInfo, ModuleInfo, TensorInfo, TensorStats


class TorchEngine:
    """Load a Transformers model and expose its static structure on demand."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._modules: dict[str, Any] = {}
        self._info: ModelInfo | None = None
        self._torch: Any | None = None

    def load(self, model: str | Path) -> ModelInfo:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise EngineError(
                "Model inspection needs the optional torch dependencies; "
                "install sememe[torch]."
            ) from exc

        identifier = str(model)
        if not identifier:
            raise EngineError("Choose a model directory or Hub id.")
        if isinstance(model, Path) and not model.is_dir():
            raise EngineError(f"Model directory does not exist: {model}")

        try:
            loaded = AutoModel.from_pretrained(
                identifier, dtype="auto", trust_remote_code=False
            )
            loaded.eval()
            modules = dict(loaded.named_modules())
            info = ModelInfo(
                class_name=type(loaded).__name__,
                config=loaded.config.to_dict(),
                modules=tuple(
                    ModuleInfo(
                        path=path,
                        class_name=type(module).__name__,
                        params=tuple(
                            _tensor_info(name, tensor)
                            for name, tensor in module.named_parameters(recurse=False)
                        ),
                        buffers=tuple(
                            _tensor_info(name, tensor)
                            for name, tensor in module.named_buffers(recurse=False)
                        ),
                    )
                    for path, module in modules.items()
                ),
            )
        except Exception as exc:
            raise EngineError(f"Could not load and inspect model: {exc}") from exc

        # Publish a complete snapshot only after loading and enumeration succeed.
        self._torch = torch
        self._model = loaded
        self._modules = modules
        self._info = info
        return info

    def model_info(self) -> ModelInfo:
        if self._info is None:
            raise EngineError("No model is loaded.")
        return self._info

    def param_stats(self, module: str, tensor: str) -> TensorStats:
        if self._model is None or self._torch is None:
            raise EngineError("No model is loaded.")
        owner = self._modules.get(module)
        if owner is None:
            raise EngineError(f"Unknown module: {module}")
        matches = (
            value
            for name, value in (
                *owner.named_parameters(recurse=False),
                *owner.named_buffers(recurse=False),
            )
            if name == tensor
        )
        value = next(matches, None)
        if value is None:
            raise EngineError(f"Module {module or '<root>'} has no tensor {tensor}.")
        if value.device.type == "meta":
            raise EngineError("This tensor has no materialized values (meta device).")
        if value.is_complex() or value.is_sparse:
            raise EngineError("Statistics for complex or sparse tensors are not supported yet.")
        if value.numel() == 0:
            raise EngineError("This tensor has no values.")

        try:
            with self._torch.inference_mode():
                return _stats(value.detach(), self._torch)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"Could not compute tensor statistics: {exc}") from exc


def _tensor_info(name: str, tensor: Any) -> TensorInfo:
    return TensorInfo(
        name=name,
        shape=tuple(int(size) for size in tensor.shape),
        dtype=str(tensor.dtype),
        numel=tensor.numel(),
        bytes=tensor.numel() * tensor.element_size(),
        device=str(tensor.device),
    )


def _stats(tensor: Any, torch: Any) -> TensorStats:
    """Compute exact reductions with bounded temporary memory, then 16 bins."""
    flat = tensor.reshape(-1)
    chunk_size = 1_000_000
    count = flat.numel()
    minimum = math.inf
    maximum = -math.inf
    total = 0.0
    total_squares = 0.0
    zero_count = 0

    for start in range(0, count, chunk_size):
        chunk = flat[start : start + chunk_size].to(dtype=torch.float32)
        if not bool(torch.isfinite(chunk).all()):
            raise EngineError("This tensor contains non-finite values.")
        minimum = min(minimum, float(chunk.min()))
        maximum = max(maximum, float(chunk.max()))
        total += float(chunk.sum(dtype=torch.float64))
        total_squares += float((chunk * chunk).sum(dtype=torch.float64))
        zero_count += int(torch.count_nonzero(chunk == 0))

    mean = total / count
    variance = max(0.0, total_squares / count - mean * mean)
    histogram = [0] * 16
    if minimum == maximum:
        histogram[0] = count
    else:
        for start in range(0, count, chunk_size):
            chunk = flat[start : start + chunk_size].to(dtype=torch.float32)
            bins = torch.histc(chunk, bins=16, min=minimum, max=maximum)
            histogram = [old + int(new) for old, new in zip(histogram, bins)]

    return TensorStats(
        min=minimum,
        max=maximum,
        mean=mean,
        std=math.sqrt(variance),
        l2_norm=math.sqrt(total_squares),
        zero_fraction=zero_count / count,
        histogram=tuple(histogram),
    )
