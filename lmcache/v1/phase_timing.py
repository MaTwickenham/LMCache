# SPDX-License-Identifier: Apache-2.0
"""Lightweight request-phase timing recorder for local benchmarks."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from lmcache.logging import init_logger

logger = init_logger(__name__)

_PHASE_TIMING_PATH = os.getenv("LMCACHE_PHASE_TIMING_PATH")
_PHASE_TIMING_LOCK = threading.Lock()


def phase_timing_enabled() -> bool:
    return bool(_PHASE_TIMING_PATH)


def record_phase(req_id: str | None, phase: str, duration_s: float, **extra: Any) -> None:
    """Append one phase timing record to a JSONL file if enabled."""

    if not req_id or not _PHASE_TIMING_PATH:
        return

    record: dict[str, Any] = {
        "ts": time.time(),
        "req_id": str(req_id),
        "phase": str(phase),
        "duration_s": float(duration_s),
    }
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            record[key] = value
        else:
            record[key] = str(value)

    line = json.dumps(record, sort_keys=True)
    try:
        with _PHASE_TIMING_LOCK:
            with open(_PHASE_TIMING_PATH, "a", encoding="utf-8") as file:
                file.write(line)
                file.write("\n")
    except OSError as exc:
        logger.warning("Failed to write LMCache phase timing record: %s", exc)
