"""The one interface between the engine and the UI.

The UI imports only this module. An engine runs the model; the UI asks it
questions through `Engine` and gets plain dataclasses back. Today both live in
one process. If the model ever runs on another machine, a socket client that
implements `Engine` slots in here and neither side changes.

Cheap first, expensive on request. `model_info` reports every module's class
and the shape, dtype and size of each tensor it owns, which needs metadata
only. `param_stats` reads a tensor's values, so the UI calls it for the
selected module, never for the whole model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class TensorInfo:
    """One parameter or buffer. No values are read to produce it."""

    name: str  # attribute name on its module, e.g. "weight"
    shape: tuple[int, ...]
    dtype: str  # the backend's own name, e.g. "torch.bfloat16"
    numel: int
    bytes: int
    device: str  # e.g. "cpu", "cuda:0", "meta"


@dataclass(frozen=True)
class ModuleInfo:
    """One module and the tensors it owns directly, not its children's."""

    path: str  # dotted path from the root; the root itself is ""
    class_name: str  # e.g. "Qwen3_5Attention", "Linear"
    params: tuple[TensorInfo, ...] = ()
    buffers: tuple[TensorInfo, ...] = ()
    description: str = ""  # backend module.extra_repr(), not an inferred role

    @property
    def own_param_count(self) -> int:
        return sum(p.numel for p in self.params)


@dataclass(frozen=True)
class ModelInfo:
    """A loaded model's static description."""

    class_name: str  # e.g. "Qwen3_5ForConditionalGeneration"
    config: dict[str, Any] = field(default_factory=dict)  # model.config.to_dict()
    modules: tuple[ModuleInfo, ...] = ()  # depth-first, named_modules() order

    def module(self, path: str) -> ModuleInfo | None:
        return next((m for m in self.modules if m.path == path), None)

    def children(self, path: str) -> list[ModuleInfo]:
        """Direct children of the module at `path`."""
        depth = 0 if path == "" else path.count(".") + 1
        prefix = "" if path == "" else path + "."
        return [
            m for m in self.modules
            if m.path.startswith(prefix) and m.path != path and m.path.count(".") == depth
        ]

    def subtree_param_count(self, path: str) -> int:
        """Parameters of the module at `path` and everything under it."""
        prefix = "" if path == "" else path + "."
        return sum(m.own_param_count for m in self.modules if m.path == path or m.path.startswith(prefix))

    @property
    def param_count(self) -> int:
        return sum(m.own_param_count for m in self.modules)


@dataclass(frozen=True)
class TensorStats:
    """Value statistics of one tensor. Computing them reads every element."""

    min: float
    max: float
    mean: float
    std: float
    l2_norm: float
    zero_fraction: float  # 0.0 to 1.0
    histogram: tuple[int, ...]  # counts in equal-width bins over [min, max]


@dataclass(frozen=True)
class LoadEvent:
    """One measured step of a load. A bar is drawn only when `total` is known,
    and its fraction always means `done` of `total` `unit` — never elapsed time.

    phase:   "import", "config", "install" or "inspect"
    message: what happened, in words, e.g. "parameters installed"
    done/total/unit: the measured basis, e.g. 120 of 473 "parameters"
    item:    the parameter or tensor this step concerned, when the loader says
    elapsed: seconds since the load started
    """

    phase: str
    message: str
    elapsed: float
    done: int | None = None
    total: int | None = None
    unit: str = ""
    item: str = ""
    finished: bool = False  # the phase is complete (its last event)


Progress = Callable[[LoadEvent], None]


@dataclass(frozen=True)
class RunSettings:
    """What the caller asks for. The record keeps these plus what was actually used.

    Bounds are checked before any forward pass and rejected visibly, never
    silently truncated: 1 <= top_k <= MAX_TOP_K, input tokens <= max_input_tokens
    <= MAX_INPUT_TOKENS_CEILING, and the raw prompt <= MAX_PROMPT_CHARS characters
    (checked before tokenizing).
    """

    top_k: int = 10
    max_input_tokens: int = 4096


MAX_TOP_K = 100
MAX_INPUT_TOKENS_CEILING = 32_768  # max_input_tokens may not be set above this
MAX_PROMPT_CHARS = 262_144  # raw text guard, checked before tokenizing


@dataclass(frozen=True)
class Token:
    """One token as the tokenizer produced it: its id and its exact decoded text."""

    id: int
    text: str


@dataclass(frozen=True)
class Candidate:
    """One next-token candidate at the final input position."""

    id: int
    text: str
    probability: float  # softmax over the full vocabulary, computed in float32
    logit: float


@dataclass(frozen=True)
class WatchSpec:
    """Observe a module in model_info's namespace; no intervention."""

    module: str
    where: str = "output"  # input | output


@dataclass(frozen=True)
class WatchBudget:
    """Hard-bounded summaries: first N tokens, at most one invocation per target.

    A token exceeding max_features is skipped, never sampled silently. No raw
    activations are retained. Limits may be lowered, not raised beyond defaults.
    """

    max_watches: int = 8
    max_tokens: int = 128
    max_features: int = 65536


@dataclass(frozen=True)
class RunResult:
    """One controlled forward pass: exact input, top next tokens, and how it ran.

    No generation: a single forward over the prompt, scored at its last position.
    `used` records the settings and runtime actually in effect (device, dtype,
    tokenizer options), never the ones merely requested.
    """

    run_id: str
    prompt: str
    tokens: tuple[Token, ...]
    candidates: tuple[Candidate, ...]
    settings: RunSettings
    used: dict[str, Any]
    model: dict[str, Any]  # identity of the loaded model: class, ref, config digest, versions
    timing: dict[str, float]  # seconds per stage
    watches: tuple[WatchSpec, ...] = ()
    watch_budget: WatchBudget = field(default_factory=WatchBudget)
    captures: tuple[dict[str, Any], ...] = ()
    record_path: str | None = None  # None when nothing was saved; see record_error
    record_error: str | None = None  # why saving the record failed, if it did


class RunFailed(Exception):
    """A run attempt that did not produce a result: failed, cancelled or rejected.
    Carries the attempt's id and where its record was written (if it was)."""

    def __init__(self, message: str, run_id: str, status: str, record_path: str | None,
                 record_error: str | None = None) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.status = status  # "failed" | "cancelled" | "rejected"
        self.record_path = record_path
        self.record_error = record_error


class EngineError(RuntimeError):
    """The engine could not answer. The message is safe to show in the UI."""


class Engine(Protocol):
    """What the UI may ask of a model. Every method may block; the UI calls
    them from a worker thread, never from its event loop."""

    def load(self, model: str | Path, progress: Progress | None = None,
             cancelled: Callable[[], bool] | None = None) -> ModelInfo:
        """Load a model (a local snapshot directory or a Hub id) and describe it.
        Loading never runs the model. `progress` receives measured LoadEvents
        from the worker thread that calls load. Every phase opens with an
        event before its work starts, so a failure is attributed to the phase
        it happened in. `cancelled` is checked before expensive steps; a
        superseded load stops there rather than allocating a model."""
        ...

    def model_info(self) -> ModelInfo:
        """The loaded model's description. Raises EngineError if none is loaded."""
        ...

    def param_stats(self, module: str, tensor: str) -> TensorStats:
        """Statistics of one tensor, e.g. ("model.layers.3.mlp.up_proj", "weight")."""
        ...

    def run(self, prompt: str, settings: RunSettings,
            cancelled: Callable[[], bool] | None = None, *,
            watches: tuple[WatchSpec, ...] = (), watch_budget: WatchBudget = WatchBudget()) -> RunResult:
        """One forward pass over `prompt`, scored at the last position. Never
        generates. `cancelled` is checked at the run's boundaries only: a forward
        already executing is not interrupted. Watches observe input/output module
        boundaries in model_info paths. Captures report unsupported shapes and
        budget limits explicitly; no raw tensors or interventions are supported."""
        ...
