from __future__ import annotations

import gc
import os
import time
from typing import Callable

MEMORY_LIMIT_BYTES = int(os.getenv("MODEL_CACHE_MEMORY_LIMIT_MB", "5120")) * 1024 * 1024
TARGET_BYTES = int(MEMORY_LIMIT_BYTES * 0.9)
IDLE_TTL_SECONDS = int(os.getenv("MODEL_CACHE_IDLE_TTL_SECONDS", "900"))

_last_used: dict[str, float] = {}


def touch(model_name: str) -> None:
    _last_used[model_name] = time.time()


def current_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        return 0
    return 0


def _candidate_unloaders() -> dict[str, Callable[[], None]]:
    import one_lines
    import one_lines_parallel
    import one_lines_reticular_one_color
    import several_globules
    import several_globules_asymmetrical
    import several_lines_parallel
    import several_lines_reticular

    return {
        "one_lines": one_lines.clear_model,
        "one_lines_parallel": one_lines_parallel.clear_model,
        "one_lines_reticular_one_color": one_lines_reticular_one_color.clear_model,
        "several_globules": several_globules.clear_model,
        "several_globules_asymmetrical": several_globules_asymmetrical.clear_model,
        "several_lines_parallel": several_lines_parallel.clear_model,
        "several_lines_reticular": several_lines_reticular.clear_model,
    }


def _pick_eviction_candidate() -> str | None:
    if not _last_used:
        return None

    now = time.time()
    stale = [
        (last_used, name)
        for name, last_used in _last_used.items()
        if now - last_used >= IDLE_TTL_SECONDS
    ]
    candidates = stale or [(last_used, name) for name, last_used in _last_used.items()]
    return sorted(candidates)[0][1]


def evict_if_needed(logger) -> None:
    rss = current_rss_bytes()
    if rss <= MEMORY_LIMIT_BYTES:
        return

    unloaders = _candidate_unloaders()
    logger.warning(
        "RSS pressure detected: %.1f MiB > %.1f MiB. Starting model eviction.",
        rss / 1024 / 1024,
        MEMORY_LIMIT_BYTES / 1024 / 1024,
    )

    while rss > TARGET_BYTES:
        name = _pick_eviction_candidate()
        if name is None:
            break

        _last_used.pop(name, None)
        unloader = unloaders.get(name)
        if unloader is None:
            continue

        unloader()
        gc.collect()
        rss = current_rss_bytes()

    logger.info("RSS after eviction attempt: %.1f MiB", rss / 1024 / 1024)
