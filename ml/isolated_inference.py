from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

_EXECUTOR: ProcessPoolExecutor | None = None
_MAX_WORKERS = int(os.getenv("ISOLATED_RARE_MODEL_WORKERS", "1"))


def _run_task(task_name: str, image, mask) -> Any:
    if task_name == "several_lines_parallel_furrow":
        import several_lines_parallel_furrow

        return several_lines_parallel_furrow.main(image, mask)

    if task_name == "several_lines_reticular_asymmetric":
        import several_lines_reticular_asymmetric

        return several_lines_reticular_asymmetric.main(image, mask)

    if task_name == "several_globules_asymmetrical_melanin":
        import several_globules_asymmetrical_melanin

        return several_globules_asymmetrical_melanin.main(image, mask)

    if task_name == "several_globules_asymmetrical_other":
        import several_globules_asymmetrical_other

        return several_globules_asymmetrical_other.main(image, mask)

    raise ValueError(f"Unknown isolated inference task: {task_name}")


def _get_executor() -> ProcessPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(
            max_workers=_MAX_WORKERS,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        )
    return _EXECUTOR


def run_isolated(task_name: str, image, mask) -> Any:
    future = _get_executor().submit(_run_task, task_name, image, mask)
    return future.result()
