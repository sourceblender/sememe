"""Where a model can come from, without loading it.

Two sources: the local Hugging Face cache, and a folder on disk. Both answer
the same question — *what would we load, and can we?* — as a `ModelChoice`.
Nothing here imports torch or huggingface_hub, so the picker works on the
mock-only install. The cache is read from its documented on-disk layout:

    <hub>/models--<org>--<name>/snapshots/<revision>/config.json
    <hub>/models--<org>--<name>/refs/<branch>        (contains a revision)
    <hub>/models--<org>--<name>/blobs/<hash>          (the actual bytes)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Transformers weight files. A bare "model.bin" is deliberately absent: that
# is CTranslate2's name (faster-whisper), a format the engine cannot load.
WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin", "tf_model.h5", "flax_model.msgpack")


@dataclass(frozen=True)
class ModelChoice:
    """One thing the user could pick. `folder` is what the engine would load."""

    label: str  # "Qwen/Qwen3.5-0.8B" or a folder name
    folder: Path
    source: str  # "hf" or "disk"
    revision: str | None = None  # the snapshot hash, for "hf"
    refs: tuple[str, ...] = ()  # branches pointing at this revision, e.g. ("main",)
    size_bytes: int | None = None  # whole repo on disk, for "hf"
    problem: str | None = None  # why it cannot be loaded; None means it can

    @property
    def loadable(self) -> bool:
        return self.problem is None


def hub_cache_dir() -> Path:
    """The cache huggingface_hub itself would use, by its documented precedence."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "huggingface" / "hub"


def unreadable(err: OSError) -> str:
    return f"can't read it ({err.strerror or type(err).__name__})"


def model_problem(folder: Path) -> str | None:
    """Why `folder` is not a loadable model folder, or None if it looks like one.
    Filesystem errors become the reason, never an exception: a stale path or an
    unreadable folder is something to explain in the dialog, not a crash."""
    try:
        if not folder.is_dir():
            return "not a folder"
        config = folder / "config.json"
        if not config.is_file():
            return "no config.json"
        # is_file() follows links, so a dangling link or a folder with a weight
        # name does not count as weights.
        if not any(p.is_file() for pattern in WEIGHT_PATTERNS for p in folder.glob(pattern)):
            if (folder / "model.bin").is_file():
                return "CTranslate2 format, not Transformers"
            return "no Transformers weight files"
        try:
            declared = json.loads(config.read_text())
        except ValueError:
            return "config.json is not valid JSON"
        if not isinstance(declared, dict) or "model_type" not in declared:
            return "config.json has no model_type, so not a Transformers model"
    except OSError as err:
        return unreadable(err)
    return None


def listing_problem(folder: Path) -> str | None:
    """Why the browser can't show `folder`'s contents, or None if it can."""
    try:
        if not folder.is_dir():
            return "not a folder"
        next(folder.iterdir(), None)
    except OSError as err:
        return unreadable(err)
    return None


def _size(blobs: Path) -> int | None:
    try:
        return sum(entry.stat().st_size for entry in os.scandir(blobs) if entry.is_file())
    except OSError:
        return None


def scan_hub_cache(hub: Path | None = None) -> list[ModelChoice]:
    """Every cached model snapshot, one entry per revision, loadable ones first."""
    hub = hub or hub_cache_dir()
    choices: list[ModelChoice] = []
    try:
        repos = sorted(p for p in hub.iterdir() if p.name.startswith("models--")) if hub.is_dir() else []
    except OSError:
        return choices
    for repo in repos:
        try:
            choices.extend(_scan_repo(repo))
        except OSError as err:
            label = repo.name.removeprefix("models--").replace("--", "/")
            choices.append(ModelChoice(label, repo, "hf", problem=unreadable(err)))
    return sorted(choices, key=lambda c: (not c.loadable, c.label.lower(), c.revision or ""))


def _scan_repo(repo: Path) -> list[ModelChoice]:
    """One repo's snapshots. OSError propagates; the caller turns it into a reason."""
    label = repo.name.removeprefix("models--").replace("--", "/")
    refs: dict[str, list[str]] = {}
    # iterdir, not glob: glob returns nothing for an unreadable folder, which
    # would report "no snapshot" when the truth is "can't read it".
    for ref in (repo / "refs").iterdir() if (repo / "refs").is_dir() else ():
        try:
            refs.setdefault(ref.read_text().strip(), []).append(ref.name)
        except OSError:
            continue
    size = _size(repo / "blobs")
    snapshots = sorted((repo / "snapshots").iterdir()) if (repo / "snapshots").is_dir() else []
    if not snapshots:
        return [ModelChoice(label, repo, "hf", size_bytes=size, problem="no snapshot downloaded")]
    return [ModelChoice(label, snap, "hf", revision=snap.name, refs=tuple(sorted(refs.get(snap.name, []))),
                        size_bytes=size, problem=model_problem(snap)) for snap in snapshots]


def choice_for_path(path: Path) -> ModelChoice:
    """What picking `path` on disk means. A file stands for the folder it sits in,
    because the engine loads a model folder, never a lone weights file."""
    path = path.expanduser()
    try:
        folder = path if path.is_dir() else path.parent
    except OSError:
        folder = path.parent
    return ModelChoice(folder.name or str(folder), folder, "disk", problem=model_problem(folder))


def human_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"
