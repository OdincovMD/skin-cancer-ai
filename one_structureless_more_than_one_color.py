import cv2
import numpy as np
import pandas as pd
import os
import requests
import base64
from joblib import load

clf = load('weight/one_structureless_more than 1.joblib')


def clahe_filter(img, limit: float, grid: tuple):
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
    :return: список признаков (среднее значение и стандартное отклонение зеленого, синего, красного и серого цветов)
             для каждой бесструктурной области на изображении
    """
    result_mask = segmentation(image, mask_of_les)
    b, g, r = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = cv2.meanStdDev(cv2.merge((b, g, r, gray)), mask=result_mask)
    return features[0].flatten().tolist() + features[1].flatten().tolist()


def classify_image(img, mask) -> str:
    """
    Классификация изображения.
    :param img: изображение для классификации
    :param mask: маска изображения
    :return: предсказанная метка
    """

    features = feature_from_im(img, mask)
    df = pd.DataFrame([features])
    pred = clf.predict(df)

    if pred[0] == 1:
        return 'brown'
    elif pred[0] == 0:
        return 'red'
    else:
        return 'yellow'


def main(img, mask):
    """
    Вывод метки изображения по пути.
    :param im: изображение
    :return: метка изображения
    """
    return classify_image(img, mask)


# if __name__ == '__main__':
#     image_path = "26.jpg"
#     img = cv2.imread(image_path)

#     rf = Roboflow(api_key="GmJT3lC4NInRGZJ2iEit")
#     project = rf.workspace("neo-dmsux").project("neo-v6wzn")
#     model = project.version(2).model

#     data = model.predict("26.jpg").json()
#     width = data['predictions'][0]['image']['width']
#     height = data['predictions'][0]['image']['height']

#     encoded_mask = data['predictions'][0]['segmentation_mask']
#     mask_bytes = base64.b64decode(encoded_mask)
#     mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
#     mask_image = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
#     mask = np.where(mask_image == 1, 255, mask_image)
#     mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

#     result = main(img, mask)
#     print(result)