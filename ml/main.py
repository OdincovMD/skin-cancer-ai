import os
import warnings
warnings.filterwarnings("ignore")
from enum import Enum
from dataclasses import dataclass
from typing import List

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException

# Локальные модули
from log import Logger

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
import several_lines_reticular_asymmetric
import several_lines_parallel_furrow
import several_lines_radial_pereferic  # не реализуется узел выше: несколько признаков -> линни -> радиальные

# Модули для обработки "several_circles"
import several_circles

# Модули для обработки "several_dots"
import several_dots

# Модули для обработки "several_globules"
import several_globules
import several_globules_asymmetrical
import several_globules_asymmetrical_other
import several_globules_asymmetrical_melanin

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

    def classify(self, image_path: str) -> ClassificationResult:
        image = cv2.imread(image_path)
        mask = log.log_function_entry_exit(mask_builder.main)(image_path)

        feature_type = self._determine_feature_type(image, mask)

        structure, properties = (
            self._process_single_feature(image, mask)
            if feature_type == FeatureType.SINGLE
            else self._process_multiple_features(image, mask)
        )

        final_class = self._get_final_classification(image)

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

        if properties[0] == LineType.RETICULAR.value:
            properties.append(log.log_function_entry_exit(one_lines_reticular.main)(image))

            if properties[1] == CountColor.ONE.value:
                properties.append(log.log_function_entry_exit(one_lines_reticular_one_color.main)(image))
            else:
                properties.append(log.log_function_entry_exit(one_lines_reticular_several_colors.main)(image))

        elif properties[0] == LineType.BRANCHED.value:
            properties.append(log.log_function_entry_exit(one_lines_branched.main)(image, mask))

        elif properties[0] == LineType.PARALLEL.value:
            properties.append(log.log_function_entry_exit(one_lines_parallel.main)(image))

        return properties

    @staticmethod
    def _process_single_dots(image, mask) -> List[str]:
        return [log.log_function_entry_exit(one_dots.main)(image, mask)]

    @staticmethod
    def _process_multiple_globules(image, mask) -> List[str]:
        properties = [log.log_function_entry_exit(several_globules.main)(image, mask)]

        if properties[0] == Symmetry.ASYMMETRIC.value:
            properties.append(log.log_function_entry_exit(several_globules_asymmetrical.main)(image))

            if properties[1] == PigmentType.MELANIN.value:
                properties.append(log.log_function_entry_exit(several_globules_asymmetrical_melanin.main)(image, mask))
            else:
                properties.append(log.log_function_entry_exit(several_globules_asymmetrical_other.main)(image, mask))

        return properties

    @staticmethod
    def _process_multiple_lines(image, mask) -> List[str]:
        properties = [log.log_function_entry_exit(several_lines.main)(image, mask)]

        if properties[0] == LineType.PARALLEL.value:
            properties.append(log.log_function_entry_exit(several_lines_parallel.main)(image))

            if properties[1] == "Борозды":
                properties.append(log.log_function_entry_exit(several_lines_parallel_furrow.main)(image, mask))

        elif properties[0] == LineType.RETICULAR.value:
            properties.append(log.log_function_entry_exit(several_lines_reticular.main)(image))

            if properties[1] == Symmetry.ASYMMETRIC.value:
                properties.append(log.log_function_entry_exit(several_lines_reticular_asymmetric.main)(image, mask))

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


app = FastAPI()
classifier = ImageClassifier()


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        result = classifier.classify(file_path)
        logger.info(f"Файл {file.filename} успешно обработан: {result}")
        return result.__dict__

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при обработке файла")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Файл {file_path} удален")

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)