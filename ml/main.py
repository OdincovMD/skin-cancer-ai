import os
import warnings
warnings.filterwarnings("ignore")
from enum import Enum
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Локальные модули
from log import Logger
import isolated_inference
import model_cache

# Основные модули
import one_several
import mask_builder

# Модули для обработки "one"
import one
import one_lines
import one_lines_reticular
import one_lines_branched
import one_lines_parallel
import one_lines_reticular_one_color
import one_lines_reticular_several_colors

# Модули для обработки "one_globules"
import one_globules
import one_globules_one_color
import one_globules_several_colors
import one_globules_several_colors_melanin

# Модули для обработки "one_structureless"
import one_structureless
import one_structureless_one_color
import one_structureless_more_than_one_color

# Модули для обработки "one_dots"
import one_dots

# Модули для обработки "several"
import several
import several_lines
import several_lines_parallel
import several_lines_reticular
import several_lines_radial_pereferic  # не реализуется узел выше: несколько признаков -> линни -> радиальные

# Модули для обработки "several_circles"
import several_circles

# Модули для обработки "several_dots"
import several_dots

# Модули для обработки "several_globules"
import several_globules
import several_globules_asymmetrical

# Финальный модуль
import final

# Инициализация логгера
log = Logger(__name__)
logger = log.logger


class FeatureType(Enum):
    SINGLE = "Один признак"
    MULTIPLE = "Несколько признаков"


class Structure(Enum):
    STRUCTURELESS = "Бесструктурная область"
    GLOBULES = "Комки"
    LINES = "Линии"
    DOTS = "Точки"
    CIRCLES = "Круги"
    PSEUDOPODIA = "Псевдоподии"


class LineType(Enum):
    CURVED = "Изогнутые"
    PARALLEL = "Параллельные"
    RETICULAR = "Ретикулярные"
    BRANCHED = "Разветвленные"


class CountColor(Enum):
    ONE = "Один цвет"
    MANY = "Несколько цветов"


class PigmentType(Enum):
    MELANIN = "Меланин"
    OTHER = "Другой пигмент"


class Symmetry(Enum):
    SYMMETRIC = "Симметричные"
    ASYMMETRIC = "Асимметричные"


@dataclass
class ClassificationResult:
    feature_type: FeatureType
    structure: Structure
    properties: List[str]
    final_class: str



class ImageClassifier:
    final_classes = {
        'Angioma': 'Ангиома (доброкачественная сосудистая опухоль)',
        'BCC': 'Базальноклеточная карцинома',
        'DF': 'Дерматофиброма (доброкачественное образование)',
        'Melanoma': 'Меланома',
        'Nevus': 'Невус (родинка)',
        'SCC': 'Плоскоклеточная карцинома',
        'SebK': 'Себорейный кератоз (доброкачественное образование)'
    }

    def build_mask(self, image_path: str):
        return log.log_function_entry_exit(mask_builder.main)(image_path)

    def classify(self, image_path: str, mask=None) -> ClassificationResult:
        image = cv2.imread(image_path)
        if mask is None:
            mask = self.build_mask(image_path)

        feature_type = self._determine_feature_type(image, mask)

        structure, properties = (
            self._process_single_feature(image, mask)
            if feature_type == FeatureType.SINGLE
            else self._process_multiple_features(image, mask)
        )

        final_class = self._get_final_classification(image)
        model_cache.evict_if_needed(logger)

        return ClassificationResult(feature_type, structure, properties, final_class)
    
    @staticmethod
    def _determine_feature_type(image, mask) -> FeatureType:
        feature_type_pred = log.log_function_entry_exit(one_several.main)(image, mask)
        return FeatureType.SINGLE if feature_type_pred == FeatureType.SINGLE.value else FeatureType.MULTIPLE

    def _process_single_feature(self, image, mask) -> tuple[Structure, List[str]]:
        structure_pred = log.log_function_entry_exit(one.main)(image)

        structure_map = {
            Structure.STRUCTURELESS.value: self._process_structureless,
            Structure.GLOBULES.value: self._process_single_globules,
            Structure.LINES.value: self._process_single_lines,
            Structure.DOTS.value: self._process_single_dots
        }

        process_func = structure_map.get(structure_pred)
        if process_func:
            return Structure(structure_pred), process_func(image, mask)

        return Structure.PSEUDOPODIA, ["Продолжение ветки в разработке"]

    def _process_multiple_features(self, image, mask) -> tuple[Structure, List[str]]:
        structure_pred = log.log_function_entry_exit(several.main)(image)

        structure_map = {
            Structure.GLOBULES.value: self._process_multiple_globules,
            Structure.CIRCLES.value: self._process_multiple_circles,
            Structure.LINES.value: self._process_multiple_lines,
            Structure.DOTS.value: self._process_multiple_dots
        }

        process_func = structure_map.get(structure_pred, lambda img, msk: [])

        return Structure(structure_pred), process_func(image, mask)

    @staticmethod
    def _process_structureless(image, mask) -> List[str]:
        properties = [log.log_function_entry_exit(one_structureless.main)(image, mask)]
        
        if properties[0] == CountColor.ONE.value:
            properties.append(log.log_function_entry_exit(one_structureless_one_color.main)(image, mask))
        else:
            properties.append(log.log_function_entry_exit(one_structureless_more_than_one_color.main)(image, mask))

        return properties
    
    @staticmethod
    def _process_single_globules(image, mask) -> List[str]:
        properties = [log.log_function_entry_exit(one_globules.main)(image)]

        if properties[0] == CountColor.ONE.value:
            properties.append(log.log_function_entry_exit(one_globules_one_color.main)(image))
        else:
            pigment_type = log.log_function_entry_exit(one_globules_several_colors.main)(image)
            properties.append(pigment_type)

            if pigment_type == PigmentType.MELANIN.value:
                properties.append(log.log_function_entry_exit(one_globules_several_colors_melanin.main)(image))

        return properties

    @staticmethod
    def _process_single_lines(image, mask) -> List[str]:
        properties = [log.log_function_entry_exit(one_lines.main)(image)]
        model_cache.touch("one_lines")

        if properties[0] == LineType.RETICULAR.value:
            properties.append(log.log_function_entry_exit(one_lines_reticular.main)(image))

            if properties[1] == CountColor.ONE.value:
                properties.append(log.log_function_entry_exit(one_lines_reticular_one_color.main)(image))
                model_cache.touch("one_lines_reticular_one_color")
            else:
                properties.append(log.log_function_entry_exit(one_lines_reticular_several_colors.main)(image))

        elif properties[0] == LineType.BRANCHED.value:
            properties.append(log.log_function_entry_exit(one_lines_branched.main)(image, mask))

        elif properties[0] == LineType.PARALLEL.value:
            properties.append(log.log_function_entry_exit(one_lines_parallel.main)(image))
            model_cache.touch("one_lines_parallel")

        return properties

    @staticmethod
    def _process_single_dots(image, mask) -> List[str]:
        return [log.log_function_entry_exit(one_dots.main)(image, mask)]

    @staticmethod
    def _process_multiple_globules(image, mask) -> List[str]:
        properties = [log.log_function_entry_exit(several_globules.main)(image, mask)]
        model_cache.touch("several_globules")

        if properties[0] == Symmetry.ASYMMETRIC.value:
            properties.append(log.log_function_entry_exit(several_globules_asymmetrical.main)(image))
            model_cache.touch("several_globules_asymmetrical")

            if properties[1] == PigmentType.MELANIN.value:
                properties.append(
                    log.log_function_entry_exit(isolated_inference.run_isolated)(
                        "several_globules_asymmetrical_melanin",
                        image,
                        mask,
                    )
                )
            else:
                properties.append(
                    log.log_function_entry_exit(isolated_inference.run_isolated)(
                        "several_globules_asymmetrical_other",
                        image,
                        mask,
                    )
                )

        return properties

    @staticmethod
    def _process_multiple_lines(image, mask) -> List[str]:
        properties = [log.log_function_entry_exit(several_lines.main)(image, mask)]

        if properties[0] == LineType.PARALLEL.value:
            properties.append(log.log_function_entry_exit(several_lines_parallel.main)(image))
            model_cache.touch("several_lines_parallel")

            if properties[1] == "Борозды":
                properties.append(
                    log.log_function_entry_exit(isolated_inference.run_isolated)(
                        "several_lines_parallel_furrow",
                        image,
                        mask,
                    )
                )

        elif properties[0] == LineType.RETICULAR.value:
            properties.append(log.log_function_entry_exit(several_lines_reticular.main)(image))
            model_cache.touch("several_lines_reticular")

            if properties[1] == Symmetry.ASYMMETRIC.value:
                properties.append(
                    log.log_function_entry_exit(isolated_inference.run_isolated)(
                        "several_lines_reticular_asymmetric",
                        image,
                        mask,
                    )
                )

        return properties

    @staticmethod
    def _process_multiple_circles(image, mask) -> List[str]:
        return [log.log_function_entry_exit(several_circles.main)(image, mask)]

    @staticmethod
    def _process_multiple_dots(image, mask) -> List[str]:
        return [log.log_function_entry_exit(several_dots.main)(image, mask)]

    def _get_final_classification(self, image) -> str:
        pred = log.log_function_entry_exit(final.main)(image)
        return self.final_classes.get(pred, "Неизвестный класс")


classifier = ImageClassifier()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from model_warmup import warmup_all

    await asyncio.to_thread(warmup_all)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _persist_upload(file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    return file_path


def _normalize_mask(mask_array):
    mask_array = np.asarray(mask_array, dtype=np.uint8)
    return np.where(mask_array > 0, 255, 0).astype(np.uint8)


def _decode_uploaded_mask(mask_bytes: bytes):
    mask_array = cv2.imdecode(
        np.frombuffer(mask_bytes, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if mask_array is None:
        raise ValueError("Не удалось декодировать маску")
    return _normalize_mask(mask_array)


async def _classify_upload(
    file: UploadFile,
    mask: UploadFile | None = None,
):
    file_path = await _persist_upload(file)

    try:
        provided_mask = None
        if mask is not None:
            provided_mask = _decode_uploaded_mask(await mask.read())

        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            classifier.classify,
            file_path,
            provided_mask,
        )
        logger.info(f"Файл {file.filename} успешно обработан: {result}")
        return result.__dict__

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при обработке файла")


@app.post("/mask")
async def create_mask(file: UploadFile = File(...)):
    file_path = await _persist_upload(file)

    try:
        mask = await asyncio.get_event_loop().run_in_executor(
            executor,
            classifier.build_mask,
            file_path,
        )
        ok, encoded = cv2.imencode(".png", _normalize_mask(mask))
        if not ok:
            raise ValueError("Не удалось закодировать маску")
        return Response(content=encoded.tobytes(), media_type="image/png")
    except Exception as e:
        logger.error(f"Ошибка при создании маски для {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при создании маски")


@app.post("/classify")
async def classify_file(
    file: UploadFile = File(...),
    mask: UploadFile | None = File(default=None),
):
    return await _classify_upload(file, mask)


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    return await _classify_upload(file)

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
