import joblib
import cv2
import numpy as np
import torch

FOREST = joblib.load("weight/one_structureless")
INFO = ["Коричневый", "Красный", "Синий", "Черный"]


def preprocess_image(img: np.ndarray, target_size: tuple = (200, 200)) -> np.ndarray:
    """
    Предобработка изображения.

    Parameters
    ----------
    img : np.ndarray
        Исходное цветное трехканальное фото новообразования
    target_size : tuple
        Целевой размер изображения после обработки

    Returns
    -------
    np.ndarray
        Обработанное изображение
    """
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    crop_size = min(width, height)
    half_crop_size = crop_size // 2

    img_cropped = img[center_y - half_crop_size:center_y + half_crop_size,
                      center_x - half_crop_size:center_x + half_crop_size]
    img_resized = cv2.resize(img_cropped, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(threshold)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)

    return img_masked


def extract_features(img_bgr: np.ndarray, img_hsv: np.ndarray) -> np.ndarray:
    """
    Извлечение признаков из изображения.

    Parameters
    ----------
    img_bgr : np.ndarray
        Изображение в формате BGR
    img_hsv : np.ndarray
        Изображение в формате HSV

    Returns
    -------
    np.ndarray
        Массив извлеченных признаков
    """
    img_bgr = torch.from_numpy(img_bgr.transpose(2, 0, 1)).double()
    masked_bgr = np.ma.masked_equal(img_bgr, 0)
    mean_bgr = masked_bgr.mean(axis=(1, 2)).data
    balanced_bgr = mean_bgr / mean_bgr.sum()
    prop_to_red = mean_bgr[1:] / mean_bgr[0]
    prop_to_green = torch.zeros(2)
    prop_to_green[0] = mean_bgr[0] / mean_bgr[1]
    prop_to_green[1] = mean_bgr[2] / mean_bgr[1]
    prop_to_blue = mean_bgr[:-1] / mean_bgr[2]

    img_hsv = torch.from_numpy(img_hsv.transpose(2, 0, 1)).double()
    masked_hsv = np.ma.masked_equal(img_hsv, 0)
    mean_hsv = masked_hsv.mean(axis=(1, 2)).data

    features = np.hstack((mean_bgr, balanced_bgr, prop_to_red,
                          prop_to_green, prop_to_blue, mean_hsv)).round(2)
    return features


def main(img: np.ndarray) -> str:
    """
    Предсказание класса новообразования на изображении.

    Parameters
    ----------
    img : np.ndarray
        Исходное изображение новообразования

    Returns
    -------
    str
        Предсказанный класс новообразования
    """
    img_bgr = preprocess_image(img)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    features = extract_features(img_bgr, img_hsv)
    prediction = FOREST.predict([features])
    return INFO[prediction[0].tolist().index(1)]


if __name__ == "__main__":
    img = cv2.imread("26.jpg")
    result = main(img)
    print(result)