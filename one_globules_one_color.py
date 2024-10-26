import joblib
import cv2
import matplotlib.image as mpimg
import numpy as np
import torch
import warnings

# Загрузка классификаторов из файлов
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    forest_brown = joblib.load("weight/one_globules_1")
    forest_selected = joblib.load("weight/one_globules_2")


def area_of_interest(img: np.ndarray, to_hsv: bool = False) -> np.ndarray:
    """
    Возвращает обработанную область интереса изображения.
    Выбирается область размером 1500x1500 пикселей, изменяется её размер до 200x200, затем выполняется фильтрация.

    :param img: исходное изображение (трехканальное)
    :param to_hsv: флаг, указывающий, нужно ли преобразовать результат в HSV
    :return: обработанное изображение размером 200x200
    """
    height, width = 1500, 1500
    center = img.shape
    x = center[1] // 2 - width // 2
    y = center[0] // 2 - height // 2

    img = img[int(y):int(y + height), int(x):int(x + width)]
    img = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(threshold)
    res = cv2.bitwise_and(img, img, mask=mask)

    if to_hsv:
        res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    return res


def create_dataset(X_new: list, y_new: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Создаёт датасет на основе изображений и их меток, возвращает выделенные признаки и метки.

    :param X_new: список путей к изображениям
    :param y_new: список меток для изображений
    :return: массив признаков и массив меток
    """
    X, y = None, None
    for (i, image) in enumerate(X_new):
        img_bgr = area_of_interest(image, to_hsv=False)
        img_bgr = torch.tensor(img_bgr, dtype=torch.double).permute(2, 0, 1).numpy()
        masked_bgr = np.ma.masked_equal(img_bgr, 0)
        mean_bgr = masked_bgr.mean(axis=(1, 2)).data
        balanced_bgr = mean_bgr / mean_bgr.sum()
        prop_to_red = mean_bgr[1:] / mean_bgr[0]
        prop_to_green = np.array([mean_bgr[0] / mean_bgr[1], mean_bgr[2] / mean_bgr[1]])
        prop_to_blue = mean_bgr[:-1] / mean_bgr[2]

        img_hsv = area_of_interest(image, to_hsv=True)
        img_hsv = torch.tensor(img_hsv, dtype=torch.double).permute(2, 0, 1).numpy()
        masked_hsv = np.ma.masked_equal(img_hsv, 0)
        mean_hsv = masked_hsv.mean(axis=(1, 2)).data

        new_row = np.hstack((mean_bgr, balanced_bgr, prop_to_red, prop_to_green, prop_to_blue, mean_hsv)).round(2)
        new_label = np.array(y_new[i])

        if X is None:
            X = new_row
            y = new_label
        else:
            X = np.vstack((X, new_row))
            y = np.vstack((y, new_label))

    return X, y


def main(img: np.ndarray) -> str:
    """
    Основная функция для классификации изображения на основе признаков.

    :param img: изображение для классификации
    :return: предсказанная категория изображения
    """
    info = ["Желтый-белый", "Коричневый", "Красный", "Оранжевый", "Телесный", "Черный"]

    X_temp = [img]
    X_processed, _ = create_dataset(X_temp, [[]])

    prediction = forest_brown.predict([X_processed])[0].tolist()

    if prediction[1] != 1:
        prediction = forest_selected.predict([X_processed])[0].tolist()
        prediction.insert(1, 0)

    try:
        return info[prediction.index(1)]
    except ValueError:
        # Если 1 не найден в списке предсказаний, возвращаем класс с максимальной вероятностью
        max_prob_index = prediction.index(max(prediction))
        return info[max_prob_index]


if __name__ == "__main__":
    img = mpimg.imread("26.jpg")
    if img is not None:
        result = main(img)
        print(result)
    else:
        print("Ошибка: не удалось загрузить изображение")
