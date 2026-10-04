"""A stand-in engine shaped like Qwen3.5-0.8B, for running the UI without torch.

The module paths, class names and parameter shapes follow the real model as
loaded by `transformers.AutoModel` (read 2026-10-01): 24 hybrid decoder layers,
three linear-attention (Gated DeltaNet) layers to every full-attention layer,
and a 12-block vision tower. Nothing here is measured: stats are synthetic and
the UI marks anything from this engine as MOCK.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Callable

from sememe.engine.api import (Candidate, EngineError, LoadEvent, ModelInfo, ModuleInfo, Progress, RunFailed, WatchSpec, WatchBudget,
                               RunResult,
                               RunSettings, TensorInfo, TensorStats, Token)

HIDDEN = 1024
INTERMEDIATE = 3584
VOCAB = 248320
LAYERS = 24
VISION_BLOCKS = 12
VISION_HIDDEN = 768
LAYER_TYPES = tuple("full_attention" if i % 4 == 3 else "linear_attention" for i in range(LAYERS))


def _t(name: str, *shape: int) -> TensorInfo:
    numel = math.prod(shape)
    return TensorInfo(name=name, shape=shape, dtype="torch.bfloat16", numel=numel, bytes=numel * 2, device="cpu")


def _linear(path: str, out_f: int, in_f: int, bias: bool = False) -> ModuleInfo:
    params = (_t("weight", out_f, in_f),) + ((_t("bias", out_f),) if bias else ())
    return ModuleInfo(path, "Linear", params)


def _norm(path: str, size: int, cls: str = "Qwen3_5RMSNorm") -> ModuleInfo:
    return ModuleInfo(path, cls, (_t("weight", size),))


def _decoder_layer(i: int) -> list[ModuleInfo]:
    p = f"language_model.layers.{i}"
    out = [ModuleInfo(p, "Qwen3_5DecoderLayer")]
    if LAYER_TYPES[i] == "linear_attention":
        a = f"{p}.linear_attn"
        out += [
            ModuleInfo(a, "Qwen3_5GatedDeltaNet", (_t("dt_bias", 16), _t("A_log", 16))),
            ModuleInfo(f"{a}.conv1d", "Conv1d", (_t("weight", 6144, 1, 4),)),
            _norm(f"{a}.norm", 128, "Qwen3_5RMSNormGated"),
            _linear(f"{a}.out_proj", HIDDEN, 2048),
            _linear(f"{a}.in_proj_qkv", 6144, HIDDEN),
            _linear(f"{a}.in_proj_z", 2048, HIDDEN),
            _linear(f"{a}.in_proj_b", 16, HIDDEN),
            _linear(f"{a}.in_proj_a", 16, HIDDEN),
        ]
    else:
        a = f"{p}.self_attn"
        out += [
            ModuleInfo(a, "Qwen3_5Attention"),
            _linear(f"{a}.q_proj", 4096, HIDDEN),
            _linear(f"{a}.k_proj", 512, HIDDEN),
            _linear(f"{a}.v_proj", 512, HIDDEN),
            _linear(f"{a}.o_proj", HIDDEN, 2048),
            _norm(f"{a}.q_norm", 256),
            _norm(f"{a}.k_norm", 256),
        ]
    out += [
        ModuleInfo(f"{p}.mlp", "Qwen3_5MLP"),
        _linear(f"{p}.mlp.gate_proj", INTERMEDIATE, HIDDEN),
        _linear(f"{p}.mlp.up_proj", INTERMEDIATE, HIDDEN),
        _linear(f"{p}.mlp.down_proj", HIDDEN, INTERMEDIATE),
        ModuleInfo(f"{p}.mlp.act_fn", "SiLUActivation"),
        _norm(f"{p}.input_layernorm", HIDDEN),
        _norm(f"{p}.post_attention_layernorm", HIDDEN),
    ]
    return out


def _vision() -> list[ModuleInfo]:
    v = "visual"
    out = [
        ModuleInfo(v, "Qwen3_5VisionModel"),
        ModuleInfo(f"{v}.patch_embed", "Qwen3_5VisionPatchEmbed"),
        ModuleInfo(f"{v}.patch_embed.proj", "Conv3d",
                   (_t("weight", VISION_HIDDEN, 3, 2, 16, 16), _t("bias", VISION_HIDDEN))),
        ModuleInfo(f"{v}.pos_embed", "Embedding", (_t("weight", 2304, VISION_HIDDEN),)),
        ModuleInfo(f"{v}.rotary_pos_emb", "Qwen3_5VisionRotaryEmbedding"),
        ModuleInfo(f"{v}.blocks", "ModuleList"),
    ]
    for i in range(VISION_BLOCKS):
        b = f"{v}.blocks.{i}"
        out += [
            ModuleInfo(b, "Qwen3_5VisionBlock"),
            ModuleInfo(f"{b}.norm1", "LayerNorm", (_t("weight", VISION_HIDDEN), _t("bias", VISION_HIDDEN))),
            ModuleInfo(f"{b}.norm2", "LayerNorm", (_t("weight", VISION_HIDDEN), _t("bias", VISION_HIDDEN))),
            ModuleInfo(f"{b}.attn", "Qwen3_5VisionAttention"),
            _linear(f"{b}.attn.qkv", 3 * VISION_HIDDEN, VISION_HIDDEN, bias=True),
            _linear(f"{b}.attn.proj", VISION_HIDDEN, VISION_HIDDEN, bias=True),
            ModuleInfo(f"{b}.mlp", "Qwen3_5VisionMLP"),
            _linear(f"{b}.mlp.linear_fc1", 3072, VISION_HIDDEN, bias=True),
            _linear(f"{b}.mlp.linear_fc2", VISION_HIDDEN, 3072, bias=True),
        ]
    out.append(ModuleInfo(f"{v}.merger", "Qwen3_5VisionPatchMerger"))
    return out


def qwen_like_model_info() -> ModelInfo:
    modules = [ModuleInfo("", "Qwen3_5Model")]
    modules += _vision()
    modules += [
        ModuleInfo("language_model", "Qwen3_5TextModel"),
        ModuleInfo("language_model.embed_tokens", "Embedding", (_t("weight", VOCAB, HIDDEN),)),
        ModuleInfo("language_model.layers", "ModuleList"),
    ]
    for i in range(LAYERS):
        modules += _decoder_layer(i)
    modules += [
        _norm("language_model.norm", HIDDEN),
        ModuleInfo("language_model.rotary_emb", "Qwen3_5TextRotaryEmbedding",
                   buffers=(_t("inv_freq", 32), _t("original_inv_freq", 32))),
    ]
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": LAYERS, "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
            "num_attention_heads": 8, "num_key_value_heads": 2, "vocab_size": VOCAB,
            "layer_types": list(LAYER_TYPES),
        },
        "vision_config": {"depth": VISION_BLOCKS, "hidden_size": VISION_HIDDEN},
    }
    return ModelInfo(class_name="Qwen3_5Model", config=config, modules=tuple(modules))


class FakeEngine:
    """Implements `Engine` with no model behind it."""

    def __init__(self, step_delay: float = 0.0, fail_at: int | None = None, fail_in: str | None = None) -> None:
        self._info: ModelInfo | None = None
        self.step_delay = step_delay  # > 0 lets `--mock` show the loading screen
        self.fail_at = fail_at  # a parameter index to fail on, for tests
        self.fail_in = fail_in  # a phase to fail at the start of ("config", "install"), for tests

    def load(self, model: str | Path, progress: Progress | None = None,
             cancelled: Callable[[], bool] | None = None) -> ModelInfo:
        """Walk the same phases as the real engine over the fake parameter list.
        Every message says MOCK: these steps measure nothing."""
        emit = progress or (lambda event: None)
        stop = cancelled or (lambda: False)
        start = time.monotonic()
        clock = lambda: time.monotonic() - start  # noqa: E731
        info = qwen_like_model_info()
        params = [(f"{m.path}.{t.name}", t) for m in info.modules for t in m.params]

        def begin(phase: str, message: str) -> None:
            if stop():
                raise EngineError(f"cancelled before {phase}")
            emit(LoadEvent(phase, message, clock()))
            if self.fail_in == phase:
                raise EngineError(f"mock failure in {phase}")

        begin("config", "reading config — MOCK")
        emit(LoadEvent("config", "config read — MOCK", clock(), finished=True))
        begin("install", "installing parameters — MOCK")
        for i, (name, _) in enumerate(params, 1):
            if self.fail_at is not None and i == self.fail_at:
                raise EngineError(f"mock failure installing {name}")
            if self.step_delay:
                time.sleep(self.step_delay)
            emit(LoadEvent("install", "parameters installed — MOCK", clock(),
                           done=i, total=len(params), unit="parameters", item=name, finished=i == len(params)))
        begin("inspect", "enumerating modules — MOCK")
        emit(LoadEvent("inspect", f"{len(info.modules)} modules enumerated — MOCK", clock(), finished=True))
        self._info = info
        return self._info

    def run(self, prompt: str, settings: RunSettings,
            cancelled: Callable[[], bool] | None = None, *,
            watches: tuple[WatchSpec, ...] = (), watch_budget: WatchBudget = WatchBudget()) -> RunResult:
        """A synthetic run: whitespace "tokens" and fixed candidates, all MOCK.
        Never written as a record, because it measures nothing."""
        if watches:
            raise RunFailed("MOCK has no real activations to observe", "mock", "rejected", None)
        not_saved = "MOCK runs are not recorded"
        if self._info is None:
            raise RunFailed("no model loaded", "mock", "rejected", None, not_saved)
        if not prompt:
            raise RunFailed("Type a prompt to run.", "mock", "rejected", None, not_saved)
        if cancelled is not None and cancelled():
            raise RunFailed("cancelled before the forward pass", "mock", "cancelled", None, not_saved)
        if self.step_delay:
            time.sleep(self.step_delay * 50)
        if cancelled is not None and cancelled():
            raise RunFailed("cancelled after the forward pass; the result was discarded", "mock", "cancelled", None,
                            not_saved)
        words = prompt.split(" ")
        tokens = tuple(Token(1000 + i, (" " if i else "") + w) for i, w in enumerate(words))
        pool = [(" Paris", 0.31), (" the", 0.12), (" a", 0.08), (":", 0.05), ("\n", 0.04),
                (" located", 0.03), (" known", 0.02), (" one", 0.02), (",", 0.01), (" in", 0.01)]
        k = max(1, min(settings.top_k, len(pool)))
        candidates = tuple(Candidate(2000 + i, text, p, math.log(p)) for i, (text, p) in enumerate(pool[:k]))
        return RunResult(run_id="mock", prompt=prompt, tokens=tokens, candidates=candidates, settings=settings,
                         used={"engine": "MOCK", "top_k": k}, model={"class": "FakeEngine — MOCK"},
                         timing={"forward": 0.0}, record_path=None)

    def close(self) -> None:
        self._info = None

    def model_info(self) -> ModelInfo:
        if self._info is None:
            raise EngineError("no model loaded")
        return self._info

    def param_stats(self, module: str, tensor: str) -> TensorStats:
        info = self.model_info().module(module)
        target = next((t for t in (info.params + info.buffers) if t.name == tensor), None) if info else None
        if target is None:
            raise EngineError(f"no tensor {tensor!r} on module {module!r}")
        rng = random.Random(f"{module}.{tensor}")
        std = 0.02 if target.name == "weight" and len(target.shape) > 1 else 0.3
        mean = rng.uniform(-0.002, 0.002) if std == 0.02 else 1.0
        hist = tuple(int(1000 * math.exp(-((b - 7.5) ** 2) / 8)) + rng.randint(0, 20) for b in range(16))
        return TensorStats(min=mean - 5 * std, max=mean + 5 * std, mean=mean, std=std,
                           l2_norm=std * math.sqrt(target.numel), zero_fraction=0.0, histogram=hist)
