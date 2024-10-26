import pandas as pd
from scipy import stats
import joblib
import cv2
import numpy as np

clf = joblib.load('weight/one_clods_single-color_clf.joblib')


def count_area_of_interest(img: np.ndarray) -> int:
    """
    Считает количество пикселей в области интереса изображения.

    :param img: исходное изображение (трехканальное)
    :return: количество ненулевых пикселей в изображении
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.countNonZero(gray)


def get_image_features(img: np.ndarray) -> dict:
    """
    Вычисляет признаки изображения для классификации.

    :param img: исходное изображение (трехканальное)
    :return: словарь признаков изображения
    """
    features = {}
    area_value = count_area_of_interest(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)

    for channel, color in zip([b, g, r], ['b', 'g', 'r']):
        channel_nonzero = channel[channel != 0]
        if len(channel_nonzero) == 0:
            # Если в канале нет ненулевых пикселей, добавляем нулевые значения признаков
            channel_nonzero = np.array([0])

        features.update({
            f'mean_{color}': np.mean(channel_nonzero),
            f'mean_{color}/area_value': np.mean(channel_nonzero) / area_value,
            f'std_{color}': np.std(channel_nonzero),
            f'std_{color}/area_value': np.std(channel_nonzero) / area_value,
            f'var_{color}': np.var(channel_nonzero),
            f'var_{color}/area_value': np.var(channel_nonzero) / area_value,
            f'sum_{color}': np.sum(channel_nonzero),
            f'sum_{color}/area_value': np.sum(channel_nonzero) / area_value,
            f'max_{color}': np.max(channel_nonzero),
            f'max_{color}/area_value': np.max(channel_nonzero) / area_value,
            f'min_{color}': np.min(channel_nonzero),
            f'min_{color}/area_value': np.min(channel_nonzero) / area_value,
            f'median_{color}': np.median(channel_nonzero),
            f'median_{color}/area_value': np.median(channel_nonzero) / area_value,
            f'mode_{color}': float(stats.mode(channel_nonzero, keepdims=False)[0]),
            f'mode_{color}/area_value': float(stats.mode(channel_nonzero, keepdims=False)[0] / area_value)
        })

    features.update({
        'var_area_interest': np.var(gray),
        'std/area_value': np.std(gray) / area_value,
        'std_area_interest': np.std(gray),
        'mean/area_value': np.mean(gray) / area_value,
        'mean_area_interest': np.mean(gray),
        'var_area_interest/area_value': np.var(gray) / area_value,
        'area_value': area_value
    })

    return features


def classify_image(img: np.ndarray) -> str:
    """
    Классифицирует изображение на основе заранее обученной модели.

    :param img: изображение для классификации
    :return: предсказанный ярлык ('single_color' или 'several_colors')
    """
    features = get_image_features(img)
    df = pd.DataFrame([features])
    pred = clf.predict(df)

    return 'single_color' if pred[0] == 0 else 'several_colors'


def main(img: np.ndarray) -> str:
    """
    Основная функция для классификации изображения.

    :param img: изображение для классификации
    :return: результат классификации
    """
    return classify_image(img)


if __name__ == '__main__':
    file_path = "26.jpg"  # Укажите путь к вашему изображению
    img = cv2.imread(file_path)
    if img is not None:
        result = main(img)
        print(result)
    else:
        print(f"Ошибка: не удалось загрузить изображение по пути {file_path}")
