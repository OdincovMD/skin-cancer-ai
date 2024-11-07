import joblib
import cv2
import matplotlib.image as mpimg
import numpy as np
import torch

forest_brown = joblib.load("weight/one_globules_1")
forest_selected = joblib.load("weight/one_globules_2")

def area_of_interest(img: np.ndarray, to_hsv: bool = False) -> np.ndarray:
    '''
    Функция. Вовзращает обработанное изображение img.
    Для этого происходит вырезка области размером 1500 х 1500, изменение её размеров до 200 х 200.
    Затем при помощи пороговой фильтрации на изображении выделяется новообразование.

    Parameters
    ----------
    img : np.ndarray
        Исходное цветное трехканальное фото новообразования
    to_hsv : bool
        Метка. Указывает, нужно ли преобразовывать конечное изображение из формата BGR в формат HSV

    Returns
    -------
    np.ndarray
        Изображенние 200x200x3, обработанное по указанному алгоритму.
    '''

    height, width = 1500, 1500
    center = img.shape
    x = center[1]/2 - width/2
    y = center[0]/2 - height/2

    img = img[int(y):int(y+height), int(x):int(x+width)]
    img = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(threshold)
    res = cv2.bitwise_and(img, img, mask=mask)
    # res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if to_hsv:
        res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    return res


def create_dataset(X_new: list, y_new: list) -> tuple[np.ndarray, np.ndarray]:
    '''
    Создает датасет на основе фотографий и меток к ним и возвращает этот датасет.

    Parameters
    ----------
    X_new : list
       Список, содержащий пути до всех обрабатываемых изображений
    y_new : list
       Список, содержащий метки для соответствующих изображений из списка X_new

    Returns
    -------
    np.ndarray
       Массив, содержащий выделенные признаки для каждого из изображений.
       Выделенными признаками являются:
       1) Среднее значение каждого из каналов B, G, R по всем оставшимся пикселам (всего 3 признака).
       2) Отношения среднего значения каналов B, G, R всех пикселов
          к сумме средних значений каналов B, G, R (всего 3 признака).
       3) Отношение среднего значения каналов B, G, R всех пикселов
          к каждому среднему значению двух других каналов (всего 6 признаков).
       4) Среднее значение каждого из каналов H, S, V по всем оставшимся пикселам (всего 3 признака).
       ----------------------------------------------------------------------------------
          Итог: 15 признаков.

    np.ndarray
       Массив, содержащий метки для каждого из изображений из массива X
    '''

    X = None
    y = None
    for (i, image) in enumerate(X_new):
        img_bgr = area_of_interest(image, to_hsv=False)
        img_bgr = torch.tensor(
            img_bgr, dtype=torch.double).permute(2, 0, 1).numpy()
        masked_bgr = np.ma.masked_equal(img_bgr, 0)
        mean_bgr = masked_bgr.mean(axis=(1, 2)).data
        balanced_bgr = mean_bgr / mean_bgr.sum()
        prop_to_red = mean_bgr[1:] / mean_bgr[0]
        prop_to_green = torch.zeros(2)
        prop_to_green[0] = mean_bgr[0] / mean_bgr[1]
        prop_to_green[1] = mean_bgr[2] / mean_bgr[1]
        prop_to_blue = mean_bgr[:-1] / mean_bgr[2]

        img_hsv = area_of_interest(image, to_hsv=True)
        img_hsv = torch.tensor(
            img_hsv, dtype=torch.double).permute(2, 0, 1).numpy()
        masked_hsv = np.ma.masked_equal(img_hsv, 0)
        mean_hsv = masked_hsv.mean(axis=(1, 2)).data

        new_row = np.hstack(
            (mean_bgr, balanced_bgr, prop_to_red, prop_to_green, prop_to_blue, mean_hsv)).round(2)
        new_label = np.array(y_new[i])

        if X is None:
            X = new_row
            y = new_label

        else:
            X = np.vstack((X, new_row))
            y = np.vstack((y, new_label))
    return X, y


def main(img: np.ndarray) -> str:
    info = ["Желтый-белый", "Коричневый", "Красный",
            "Оранжевый", "Телесный", "Черный"]

    X_temp = [img]
    X_processed, _ = create_dataset(X_temp, [[]])

    prediction = forest_brown.predict([X_processed])[0].tolist()

    if prediction[1] != 1:
        prediction = forest_selected.predict([X_processed])[0].tolist()
        prediction.insert(1, 0)

    try:
        return info[prediction.index(1)]
    except ValueError:
        # If 1 is not in the prediction list, return the class with the highest probability
        max_prob_index = prediction.index(max(prediction))
        print(max_prob_index)
        return info[max_prob_index]


if __name__ == "__main__":
    img = mpimg.imread("8642.jpg")
    result = main(img)
    print(result)