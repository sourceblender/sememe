"""Run records on disk: one directory per run, a bounded JSON file inside.

A record holds what a run was and what it produced: the exact tokens, the top
candidates, the settings requested and actually used, the model's identity and
the timings. It holds no tensors; captures with explicit budgets come later.
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


def write_record(result: Any) -> Path:
    """Write `result` (a RunResult) as record.json under its run id; return the path."""
    folder = runs_dir() / result.run_id
    folder.mkdir(parents=True, exist_ok=False)
    payload = dataclasses.asdict(result)
    payload.pop("record_path", None)
    payload["schema"] = "sememe.run/1"
    path = folder / "record.json"
    tmp = folder / "record.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)  # never a half-written record under the final name
    return path
