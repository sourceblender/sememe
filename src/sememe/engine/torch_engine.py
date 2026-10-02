"""PyTorch model inspection behind the UI's small Engine interface.

Importing this module does not import torch or transformers. The cockpit can
run in mock mode without the optional model dependencies installed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .api import (Candidate, EngineError, LoadEvent, ModelInfo, ModuleInfo, Progress, RunResult, RunSettings,
                  TensorInfo, TensorStats, Token)
from ..records import new_run_id, write_record


class TorchEngine:
    """Load a Transformers model and expose its static structure on demand."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._modules: dict[str, Any] = {}
        self._info: ModelInfo | None = None
        self._torch: Any | None = None
        self._full: Any | None = None  # the task model with its output head, when runnable
        self._tokenizer: Any | None = None
        self._identity: dict = {}

    def load(self, model: str | Path, progress: Progress | None = None,
             cancelled: Callable[[], bool] | None = None) -> ModelInfo:
        """Load weights and describe the model. Never runs a forward pass.

        Phases, each reported with a measured basis (see `LoadEvent`):
        import → config → install (parameters installed into the model, counted
        from transformers' own loading loop) → inspect.

        Installed is not "read from disk": transformers memory-maps safetensors,
        so a weight's bytes are paged in when something first touches them (for
        example `param_stats`). No step here claims otherwise.
        """
        emit = progress or (lambda event: None)
        start = time.monotonic()
        clock = lambda: time.monotonic() - start  # noqa: E731
        phase = "import"
        stop = cancelled or (lambda: False)

        def check() -> None:
            if stop():
                raise EngineError(f"cancelled before {phase}: a newer load replaced this one")

        try:
            emit(LoadEvent("import", "importing torch and transformers", clock()))
            try:
                import torch
                import transformers
                from transformers import AutoConfig, AutoModel
            except ImportError as exc:
                raise EngineError(
                    "Model inspection needs the optional torch dependencies; "
                    "install sememe[torch]."
                ) from exc
            emit(LoadEvent("import", "torch and transformers imported", clock(), finished=True))

            identifier = str(model)
            if not identifier:
                raise EngineError("Choose a model directory or Hub id.")
            if isinstance(model, Path) and not model.is_dir():
                raise EngineError(f"Model directory does not exist: {model}")

            phase = "config"
            check()
            emit(LoadEvent("config", "reading config", clock()))
            config = AutoConfig.from_pretrained(identifier, trust_remote_code=False)
            emit(LoadEvent("config", f"config read: {type(config).__name__}", clock(), finished=True))

            phase = "install"
            emit(LoadEvent("install", "installing parameters", clock()))
            with _install_events(check, lambda done, total, name: emit(LoadEvent(
                    "install", "parameters installed", clock(), done=done, total=total,
                    unit="parameters", item=name, finished=done == total))) as seen:
                # The task model (with its output head) when transformers knows the
                # architecture; otherwise the base model, which can be inspected
                # but not run. Inspection always looks at the base model, so
                # module addresses are the same either way.
                full_cls = _task_class(transformers, config)
                full = (full_cls or AutoModel).from_pretrained(identifier, config=config, dtype="auto",
                                                               trust_remote_code=False)
            if not seen:
                emit(LoadEvent("install", "weights installed (this transformers version reports no per-parameter "
                               "events)", clock(), finished=True))
            full.eval()
            loaded = _base_of(full)

            phase = "tokenizer"
            check()
            emit(LoadEvent("tokenizer", "loading tokenizer", clock()))
            tokenizer = None
            tokenizer_note = "no tokenizer: this transformers build has no AutoTokenizer"
            auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
            if full_cls is None:
                tokenizer_note = "no output head for this architecture, so the model can be inspected but not run"
            elif auto_tokenizer is not None:
                try:
                    tokenizer = auto_tokenizer.from_pretrained(identifier, trust_remote_code=False)
                    tokenizer_note = f"tokenizer loaded: {type(tokenizer).__name__}"
                except Exception as exc:  # a model without a tokenizer is still inspectable
                    tokenizer_note = f"no tokenizer ({exc}); the model can be inspected but not run"
            emit(LoadEvent("tokenizer", tokenizer_note, clock(), finished=True))

            phase = "inspect"
            emit(LoadEvent("inspect", "enumerating modules", clock()))
            modules = dict(loaded.named_modules())
            info = ModelInfo(
                class_name=type(loaded).__name__,
                config=loaded.config.to_dict(),
                modules=tuple(
                    ModuleInfo(
                        path=path,
                        class_name=type(module).__name__,
                        description=module.extra_repr(),
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
            emit(LoadEvent("inspect", f"{len(modules)} modules enumerated", clock(), finished=True))
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"{phase} failed: {exc}") from exc

        # Publish a complete snapshot only after loading and enumeration succeed.
        self._torch = torch
        self._model = loaded
        self._full = full if full_cls is not None else None
        self._tokenizer = tokenizer
        self._identity = _identity(identifier, full, config, torch, transformers)
        self._modules = modules
        self._info = info
        return info

    def run(self, prompt: str, settings: RunSettings,
            cancelled: Callable[[], bool] | None = None) -> RunResult:
        """One forward pass over `prompt`, scored at the final position.

        The forward itself cannot be interrupted; `cancelled` is checked before
        tokenizing, before the forward, and after it (a run cancelled after its
        forward is reported as cancelled and not recorded)."""
        stop = cancelled or (lambda: False)
        # Local references: a Close or a new load may clear the engine's fields
        # while this forward runs; the run keeps using the model it started with.
        full, tokenizer, torch, identity = self._full, self._tokenizer, self._torch, dict(self._identity)
        if full is None or tokenizer is None or torch is None:
            raise EngineError("This model can be inspected but not run: it has no output head or no tokenizer.")
        if not prompt:
            raise EngineError("Type a prompt to run.")
        if settings.top_k < 1:
            raise EngineError("top_k must be at least 1.")
        timing: dict[str, float] = {}
        if stop():
            raise EngineError("cancelled before tokenizing")
        t0 = time.monotonic()
        encoded = tokenizer(prompt, return_tensors="pt")
        ids = encoded["input_ids"]
        timing["tokenize"] = time.monotonic() - t0
        if ids.shape[-1] == 0:
            raise EngineError("The prompt tokenized to zero tokens.")
        tokens = tuple(Token(int(i), tokenizer.decode([int(i)])) for i in ids[0])
        device = next(full.parameters()).device
        if stop():
            raise EngineError("cancelled before the forward pass")
        t0 = time.monotonic()
        try:
            with torch.inference_mode():
                logits = full(**{k: v.to(device) for k, v in encoded.items()}).logits
        except Exception as exc:
            raise EngineError(f"forward failed: {exc}") from exc
        timing["forward"] = time.monotonic() - t0
        if stop():
            raise EngineError("cancelled after the forward pass; the result was discarded")
        t0 = time.monotonic()
        last = logits[0, -1].float()
        probs = torch.softmax(last, dim=-1)
        k = min(settings.top_k, probs.shape[-1])
        top_p, top_i = torch.topk(probs, k)
        candidates = tuple(Candidate(int(i), tokenizer.decode([int(i)]), float(p), float(last[int(i)]))
                           for p, i in zip(top_p, top_i))
        timing["score"] = time.monotonic() - t0
        used = {
            "device": str(device),
            "dtype": str(logits.dtype).removeprefix("torch."),
            "scored_dtype": "float32",
            "position": int(ids.shape[-1]) - 1,
            "tokenizer": type(tokenizer).__name__,
            "add_special_tokens": True,
            "chat_template": False,
            "vocab_size": int(probs.shape[-1]),
            "top_k": k,
        }
        result = RunResult(run_id=new_run_id(), prompt=prompt, tokens=tokens, candidates=candidates,
                           settings=settings, used=used, model=identity, timing=timing)
        path = write_record(result)
        return dataclasses.replace(result, record_path=str(path))

    def close(self) -> None:
        """Drop every reference to the model so its memory can be freed."""
        self._model = None
        self._full = None
        self._tokenizer = None
        self._modules = {}
        self._info = None

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


# transformers installs weights in a loop it wraps with its own `tqdm`
# (transformers.core_model_loading, "Loading weights"): one step per parameter,
# over a sized collection. We swap that name for a pass-through that yields every
# item unchanged and reports the count. It is a module global, so loads are
# serialised; a second load waits rather than racing the swap.
_INSTALL_LOCK = threading.Lock()


@contextlib.contextmanager
def _install_events(check, report):
    seen: list[int] = []
    with _INSTALL_LOCK:
        check()  # a load that waited here behind another may be superseded by now
        try:
            import transformers.core_model_loading as loading
        except ImportError:  # a transformers without this module: no events
            yield seen
            return
        original = getattr(loading, "tqdm", None)

        class PassThrough:
            def __init__(self, iterable=None, *args, **kwargs):
                self.iterable = iterable if iterable is not None else ()
                self.total = len(self.iterable) if hasattr(self.iterable, "__len__") else None

            def __iter__(self):
                for i, item in enumerate(self.iterable, 1):
                    yield item
                    seen.append(i)
                    name = item[0] if isinstance(item, tuple) and item else str(item)
                    if self.total:
                        report(i, self.total, str(name))

            def __getattr__(self, name):  # update/close/set_postfix and friends
                return lambda *a, **k: None

        if original is None:
            yield seen
            return
        loading.tqdm = PassThrough
        try:
            yield seen
        finally:
            loading.tqdm = original


def _task_class(transformers, config):
    """The class named by the config's architectures, if transformers has it and
    it carries an output head. None means: inspect only."""
    for name in getattr(config, "architectures", None) or ():
        cls = getattr(transformers, name, None)
        if cls is not None and "For" in name:
            return cls
    return None


def _base_of(full):
    """The base model inside a task model (`.model`), so inspected addresses match
    what the base class alone reports. Falls back to the model itself."""
    base = getattr(full, "model", None)
    return base if base is not None and hasattr(base, "named_modules") else full


def _identity(identifier, full, config, torch, transformers) -> dict:
    """Enough to tell later whether a record or setup belongs to this model."""
    try:
        config_json = config.to_json_string(use_diff=False)
    except Exception:
        config_json = json.dumps(getattr(config, "__dict__", {}), default=str, sort_keys=True)
    return {
        "ref": identifier,
        "class": type(full).__name__,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "torch": getattr(torch, "__version__", "?"),
        "transformers": getattr(transformers, "__version__", "?"),
    }
