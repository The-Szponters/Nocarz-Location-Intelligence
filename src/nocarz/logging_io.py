"""Thread-safe JSON-Lines request logging.

One JSON object per prediction request is appended to logs/predictions.jsonl.
JSONL is append-friendly and trivially read back with
``pd.read_json(path, lines=True)``. A module-level lock + flush keeps writes
intact under uvicorn's threadpool (run with --workers 1 for the demo).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from nocarz.features import LOGS_DIR

LOG_PATH = LOGS_DIR / "predictions.jsonl"
SCHEMA_VERSION = 1

_LOCK = threading.Lock()


def append_log(record: dict, path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
