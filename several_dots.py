import cv2
import pandas as pd
import numpy as np
from joblib import load


CLF = load('weight/several_dots.joblib')


def calc_area_of_interest(img: np.ndarray) -> np.ndarray:
    """
    Finding an area of interest and creating a mask.
    
    Paraneters:
        img (np.ndarray): The input image.
    Returns:
        np.ndarray: The image with a mask.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)
    return cv2.bitwise_and(img, img, mask=mask)


# TODO: Классификация по стандартному отклонению по каналам не очень хорошая вещь. Лучше добавить ещё фичей: рассмотреть медиану, возможно, другие цветовые пространства.
def get_image_features(img: np.ndarray) -> dict:
    """
    Transferring the image to the mask creation function and finding features for classification.
    
    Paraneters:
        img (np.ndarray): The input image.
    Returns:
        np.ndarray: Signs for classification.
    """    
    area_interest = calc_area_of_interest(img)
    b, g, r = cv2.split(area_interest)
    return {
        'std_b': np.std(b),
        'std_g': np.std(g),
        'std_r': np.std(r)
    }


def main(image: np.ndarray) -> str:
    """
    Calls the feature acquisition function and returns the predicted class.
    
    Paraneters:
        img (np.ndarray): The input image.
    Returns:
        str: The label of the predicted class: "Черный" or "Коричневый".
    """
    features = get_image_features(image)
    df = pd.DataFrame([features])
    pred = CLF.predict(df)
    return 'Черный' if pred[0] == 1 else 'Коричневый'


# Оставлено для тестирования работы модуля, при успешной проверке можно удалить 
if __name__ == '__main__':
    image = cv2.imread('26.jpg')
    print(main(image))