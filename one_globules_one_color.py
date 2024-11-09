import joblib
import cv2
import numpy as np
import torch

forest_brown = joblib.load("weight/one_globules_one_color_1")
forest_selected = joblib.load("weight/one_globules_one_color_2")

def area_of_interest(img: np.ndarray, to_hsv: bool = False) -> np.ndarray:
    """
    Returns the processed area of interest of the image.
    A 1500x1500 pixel area is selected, resized to 200x200, and then filtered.

    :param img: original image (three-channel)
    :param to_hsv: flag indicating whether to convert the result to HSV
    :return: processed 200x200 image
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
    Creates a dataset based on images and their labels, returns extracted features and labels.

    :param X_new: list of image paths
    :param y_new: list of labels for images
    :return: feature array and label array
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
    Main function for image classification based on features.

    :param img: image to classify
    :return: 'Желтый-белый', 'Коричневый', 'Красный', 'Оранжевый', 'Телесный', 'Телесный'
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
        max_prob_index = prediction.index(max(prediction))
        return info[max_prob_index]