import cv2
import pandas as pd
import numpy as np
from joblib import load


CLF = load('weight/several_dots.joblib')


# TODO: Классификация по стандартному отклонению по каналам не очень хорошая вещь. Лучше добавить ещё фичей: рассмотреть медиану, возможно, другие цветовые пространства.
def get_image_features(masked_image: np.ndarray) -> dict:
    """
    Transferring the image to the mask creation function and finding features for classification.
    
    Paraneters:
        masked_image (np.ndarray): The input image with a mask.
    Returns:
        np.ndarray: Signs for classification.
    """    
    b, g, r = cv2.split(masked_image)
    return {
        'std_b': np.std(b),
        'std_g': np.std(g),
        'std_r': np.std(r)
    }


def main(image: np.ndarray, mask: np.ndarray) -> str:
    """
    Calls the feature acquisition function and returns the predicted class.
    
    Paraneters:
        img (np.ndarray): The input image.
        mask (np.ndarray): The mask on the image that obscures everything except the mole.
    Returns:
        str: The label of the predicted class: "Черный" or "Коричневый".
    """
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    features = get_image_features(masked_image)

    df = pd.DataFrame([features])
    pred = CLF.predict(df)
    return 'Черный' if pred[0] == 1 else 'Коричневый'