import cv2
import pandas as pd
import numpy as np
from joblib import load
from typing import Dict

CLF = load('weight/several_lines_radial_edges_pereferic.joblib')
# TODO: переименовать вес в соответствии с названием модуля -> several_lines_radial_edge_pereferic.joblib
# это опять глобальная переменная в таком случае

# TODO: Рассмотреть возможность разработки дополнительных признаков для классификации, потому что это кринж
# Классификация на два класса: "белый или светло-коричневый центр" или "черный, коричневый или синий центр".
# Возможно, потребуется добавить сегментацию центра?

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the image to extract a binary mask based on background color range.

    Args:
        img (np.ndarray): The input image in BGR format.

    Returns:
        np.ndarray: A binary mask of the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = np.mean(gray[0, :]) + np.mean(gray[-1, :]) + np.mean(gray[:, 0]) + np.mean(gray[:, -1])
    bg /= (2 * (gray.shape[0] + gray.shape[1]))
    mask = cv2.inRange(gray, bg - 45, bg + 45)
    return mask


def get_image_features(img: np.ndarray) -> Dict[str, float]:
    """
    Extracts color features from the image based on the generated mask.

    Args:
        img (np.ndarray): The input image in BGR format.

    Returns:
        Dict[str, float]: A dictionary containing average RGB values.
    """
    mask = preprocess_image(img)
    r, g, b = cv2.mean(img, mask=mask)[:3]
    return {'r': r, 'g': g, 'b': b}


def main(img: np.ndarray) -> str:
    """
    Classifies the image into categories based on the color features.

    Args:
        img (np.ndarray): The input image in BGR format.

    Returns:
        str: Classification result as "BRIGHT STRUCTURELESS REGION" or "DARK STRUCTURELESS REGION".
    """
    features = get_image_features(img)
    df = pd.DataFrame([features])
    pred = CLF.predict(df)
    return 'СВЕТЛАЯ БЕССТРУКТУРНАЯ ОБЛАСТЬ' if pred == 1 else 'ТЕМНАЯ БЕССТРУКТУРНАЯ ОБЛАСТЬ'


if __name__ == "__main__":
    image_path = "26.jpg"
    image = cv2.imread(image_path)
    label = main(image)
    print(label)
