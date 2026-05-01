"""
Однократная подгрузка тяжёлых моделей при старте процесса (без прогрева на первом запросе пользователя).
"""
from __future__ import annotations

from typing import Callable, cast

import numpy as np

from log import Logger

_log = Logger("warmup").logger
_warmup_started = False
_warmup_finished = False
_warmup_results: dict[str, bool] = {}


def _model_specs() -> list[dict[str, object]]:
    import mask_builder
    import one_several
    import several
    import several_lines
    import several_lines_parallel
    import several_lines_reticular

    return [
        {
            "key": "one_several_ensemble",
            "label": "Ансамбль one/several",
            "loaded": lambda: bool(one_several._ENSEMBLE_MODELS),
        },
        {
            "key": "several_cnn",
            "label": "Классификатор several",
            "loaded": lambda: several._several_cnn is not None,
        },
        {
            "key": "several_lines",
            "label": "Модель several lines",
            "loaded": lambda: several_lines._model_several_lines is not None,
        },
        {
            "key": "several_lines_parallel",
            "label": "Модель parallel lines",
            "loaded": lambda: several_lines_parallel._model_several_lines_parallel is not None,
        },
        {
            "key": "several_lines_reticular",
            "label": "Модель reticular lines",
            "loaded": lambda: several_lines_reticular._model_several_lines_reticular is not None,
        },
        {
            "key": "mask_yolo",
            "label": "YOLO для маски",
            "loaded": lambda: mask_builder._yolo_model is not None,
        },
        {
            "key": "mask_unet",
            "label": "UNet для маски",
            "loaded": lambda: mask_builder._unet_model is not None,
        },
    ]


def _safe(step: str, fn) -> None:
    try:
        fn()
        _warmup_results[step] = True
        _log.info("Warmup OK: %s", step)
    except Exception as e:
        _warmup_results[step] = False
        _log.warning("Warmup skipped %s: %s", step, e)


def warmup_all() -> None:
    global _warmup_started, _warmup_finished
    _warmup_started = True
    _warmup_finished = False
    _warmup_results.clear()
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

    _warmup_finished = True
    _log.info("Model warmup finished")


def get_model_status() -> dict:
    models = []
    for spec in _model_specs():
        is_loaded = bool(cast(Callable[[], bool], spec["loaded"])())
        models.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "loaded": is_loaded,
            }
        )

    loaded_count = sum(1 for model in models if model["loaded"])
    total_count = len(models)

    if not _warmup_started:
        state = "not_started"
    elif not _warmup_finished:
        state = "warming"
    elif loaded_count == total_count:
        state = "ready"
    elif loaded_count > 0:
        state = "partial"
    else:
        state = "cold"

    return {
        "state": state,
        "loaded_count": loaded_count,
        "total_count": total_count,
        "models": models,
        "warmup_started": _warmup_started,
        "warmup_finished": _warmup_finished,
        "warmup_results": dict(_warmup_results),
    }
