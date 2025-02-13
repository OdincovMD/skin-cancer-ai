import cv2
import pandas as pd
from joblib import load

clf = load('weight/one_structureless.joblib')

def clahe_filter(img, limit: float = 10, grid: tuple = (6, 4)):
    """
    Увеличение контрастности изображения с помощью CLAHE.
    :param img: входное изображение
    :param limit: предел контрастности
    :param grid: количество плиток по строкам и столбцам
    :return: результат увеличения контрастности
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    return clahe.apply(l)


def segmentation(image, mask):
    """
    Сегментация изображения.
    :param image: входное изображение
    :param mask: маска изображения
    :return: сегментированная маска
    """
    equ = clahe_filter(image, 10, (6, 4))
    masked_image = cv2.bitwise_and(equ, equ, mask=mask)
    _, binar = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_and(binar, binar, mask=mask)

def feature_from_im(image, mask_of_les):
    """
    Извлечение признаков из изображения.
    :param image: входное изображение
    :param mask_of_les: маска пигментированного кожного поражения
    :return: список признаков (количество в 20 интервалах канала Hue)
    """
    result_mask = segmentation(image, mask_of_les)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    counts = cv2.calcHist([hsv], [0], result_mask, [20], [0, 180])

    return [counts.transpose()[0].tolist()]

def main(img, mask) -> str:
    """
    Классификация изображения.
    :param img: изображение для классификации
    :param mask: маска изображения
    :return: предсказанная метка "Один цвет" , "Несколько цветов"
    """

    features = feature_from_im(img, mask)
    df = pd.DataFrame([features[0],])
    pred = clf.predict(df)

    if pred[0] == 1:
        return 'Один цвет'
    else:
        return 'Несколько цветов'

