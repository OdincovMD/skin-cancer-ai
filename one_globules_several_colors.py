import cv2
import pandas as pd
from joblib import load
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    classifier = load('weight/glob_melanin_other_pigment.joblib')


def extract_color_features(img: cv2.Mat) -> dict:
    """
    Извлекает цветовые признаки из изображения, включая средние значения каналов BGR и HSV.

    :param img: исходное изображение (трехканальное)
    :return: словарь, содержащий цветовые признаки изображения
    """
    features = {}
    bgr_means = cv2.mean(img)[:3]
    hsv_means = cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[:3]

    features['mean_b'], features['mean_g'], features['mean_r'] = bgr_means
    features['mean_h'], features['mean_s'], features['mean_v'] = hsv_means

    return features


def main(img: cv2.Mat) -> str:
    """
    Основная функция для классификации изображения на основе извлечённых цветовых признаков.

    :param img: изображение для классификации
    :return: метка, определяющая класс изображения ('melanin' или 'other')
    """
    features = extract_color_features(img)
    df = pd.DataFrame([features])
    label = 'melanin' if classifier.predict(df)[0] == 1 else 'other'
    return label


if __name__ == "__main__":
    img = cv2.imread('26.jpg')
    if img is not None:
        result = main(img)
        print(result)
    else:
        print("Ошибка: не удалось загрузить изображение")
