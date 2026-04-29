"""
Однократная подгрузка тяжёлых моделей при старте процесса (без прогрева на первом запросе пользователя).
"""
from __future__ import annotations

import numpy as np

from log import Logger

_log = Logger("warmup").logger


def _safe(step: str, fn) -> None:
    try:
        fn()
        _log.info("Warmup OK: %s", step)
    except Exception as e:
        _log.warning("Warmup skipped %s: %s", step, e)


def warmup_all() -> None:
    _log.info("Model warmup started")

    import one_several

    _safe("one_several._get_ensemble", one_several._get_ensemble)

    import several

    _safe("several._get_several_cnn", several._get_several_cnn)

    import several_lines
    import several_lines_parallel
    import several_lines_reticular

    _safe("several_lines.get_model", several_lines.get_model)
    _safe("several_lines_parallel.get_model", several_lines_parallel.get_model)
    _safe("several_lines_reticular.get_model", several_lines_reticular.get_model)

    import mask_builder

    _safe("mask_builder.ensure_mask_models_warm", mask_builder.ensure_mask_models_warm)

    import final

    def _warm_final():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        final.main(img)

    _safe("final.main (hierarchical)", _warm_final)

    _log.info("Model warmup finished")
