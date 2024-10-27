import os
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
from joblib import load

# TODO: Рассмотреть возможность разработки дополнительных признаков для классификации, потому что это кринж
# Классификация на два класса: "белый или светло-коричневый центр" или "черный, коричневый или синий центр".
# Возможно, потребуется добавить сегментацию центра?

# переименовано с several_lines_radial_edges_pereferic.joblib -> several_lines_radial_edge_pereferic.joblib
MODEL_PATH = os.path.join('weight', 'several_lines_radial_edge_pereferic.joblib')


def load_model(path: str = MODEL_PATH) -> Any:
    """
    Load a model from a specified file path.

    Args:
        path (str): The file path to the model. Defaults to MODEL_PATH.

    Returns:
        Any: The loaded model object.
    """
    with open(path, 'rb') as file:
        return load(file)


_model_several_lines_reticular_or_branched_asymmetric = None


def get_model() -> Any:
    """
    Retrieve the model, loading it from the file if it has not been loaded yet.

    Returns:
        Any: The loaded model object.
    """
    global _model_several_lines_reticular_or_branched_asymmetric
    if not _model_several_lines_reticular_or_branched_asymmetric:
        _model_several_lines_reticular_or_branched_asymmetric = load_model()
    return _model_several_lines_reticular_or_branched_asymmetric


# оставлена старая функция по созданию маски, убрана из использования
# def preprocess_image(img: np.ndarray) -> np.ndarray:
#     """
#     Preprocess the image to extract a binary mask based on background color range.

#     Args:
#         img (np.ndarray): The input image in BGR format.

#     Returns:
#         np.ndarray: A binary mask of the image.
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     bg = np.mean(gray[0, :]) + np.mean(gray[-1, :]) + np.mean(gray[:, 0]) + np.mean(gray[:, -1])
#     bg /= (2 * (gray.shape[0] + gray.shape[1]))
#     mask = cv2.inRange(gray, bg - 45, bg + 45)
#     return mask

# оставлено, чтобы была функция для создания других признаков
def get_image_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extracts color features from the image based on the generated mask.

    Args:
        img (np.ndarray): The input image in BGR format.
        img (np.ndarray): The mask for image

    Returns:
        Dict[str, float]: A dictionary containing average RGB values.
    """
    r, g, b = cv2.mean(img, mask=mask)[:3]
    return {'r': r, 'g': g, 'b': b}


def main(img: np.ndarray, mask) -> str:
    """
    Classifies the image into categories based on the color features.

    Args:
        img (np.ndarray): The input image in BGR format.

    Returns:
        str: "СВЕТЛАЯ БЕССТРУКТУРНАЯ ОБЛАСТЬ" or "ТЕМНАЯ БЕССТРУКТУРНАЯ ОБЛАСТЬ".
    """
    model = get_model()
    features = get_image_features(img, mask)
    df = pd.DataFrame([features])
    pred = model.predict(df)
    return 'СВЕТЛАЯ БЕССТРУКТУРНАЯ ОБЛАСТЬ' if pred == 1 else 'ТЕМНАЯ БЕССТРУКТУРНАЯ ОБЛАСТЬ'


# оставлено, потому что модуль требует доработки
if __name__ == "__main__":
    image_path = "26.jpg"
    image = cv2.imread(image_path)
    label = main(image)
    print(label)
