"""Run records on disk: one directory per run, a bounded JSON file inside.

A record holds what a run was and what it produced: the exact tokens, the top
candidates, the settings requested and actually used, the model's identity and
the timings. Observe-only captures contain bounded per-token summaries and
explicit truncation/unsupported statuses, never raw activation tensors.
"""

from __future__ import annotations

import dataclasses
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any


def runs_dir() -> Path:
    override = os.environ.get("SEMEME_RUNS_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".local" / "share") / "sememe" / "runs"


def new_run_id() -> str:
    """Sortable by time, unique enough for one machine: 20261002T211830Z-3f9a1c."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(3)


def write_record(result: Any, status: str = "ok", **extra: Any) -> Path:
    """Write a run (a RunResult, or a dict for an attempt that produced none) as
    record.json under its run id, and return the path. Raises OSError if it
    could not be written; callers must not then report the run as saved."""
    payload = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else dict(result)
    payload.pop("record_path", None)
    payload.pop("record_error", None)
    payload.update(extra)
    payload["schema"] = "sememe.run/1"
    payload["status"] = status
    folder = runs_dir() / payload["run_id"]
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "record.json"
    tmp = folder / "record.json.tmp"
    # allow_nan=False: a NaN or infinity is never written into a record as if it
    # were a number; it raises ValueError and the caller reports "not saved".
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str, allow_nan=False))
    tmp.replace(path)  # never a half-written record under the final name
    return path
