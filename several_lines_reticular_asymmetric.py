import os
import pickle
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd

#переименовано с several_lines_reticOrBranch_asymmetric.pkl -> several_lines_reticular_asymmetric.pkl
MODEL_PATH = os.path.join('weight', 'several_lines_reticular_asymmetric.pkl')

_model_several_lines_reticular_asymmetric = None


def load_model(path: str = MODEL_PATH) -> Any:
    """
    Load a model from a specified file path.

    Args:
        path (str): The file path to the model. Defaults to MODEL_PATH.

    Returns:
        Any: The loaded model object.
    """
    with open(path, 'rb') as file:
        return pickle.load(file)



def get_model() -> Any:
    """
    Retrieve the model, loading it from the file if it has not been loaded yet.

    Returns:
        Any: The loaded model object.
    """
    global _model_several_lines_reticular_asymmetric
    if not _model_several_lines_reticular_asymmetric:
        _model_several_lines_reticular_asymmetric = load_model()
    return _model_several_lines_reticular_asymmetric


def segment_area_of_interest(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a mask to the input image and converts it to RGB color space.

    Args:
        img (np.ndarray): The original image in BGR format.
        mask (np.ndarray): The binary mask for segmentation.

    Returns:
        np.ndarray: Segmented image in RGB color space.
    """
    seg_img = cv2.bitwise_and(img, img, mask=mask)
    return cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)


def extract_characteristics(seg_img: np.ndarray) -> Dict[str, float]:
    """
    Extracts various color and texture characteristics from the segmented image in BGR and HSV

    Args:
        seg_img (np.ndarray): Segmented image in RGB.

    Returns:
        dict: A dictionary of extracted image features such as color means, max/min values, and standard deviations.
    """
    b, g, r = cv2.split(seg_img)
    mask_rgb = (b > 0) | (g > 0) | (r > 0)

    sum_rgb = np.sum(seg_img, axis=2)
    sum_rgb = sum_rgb[sum_rgb != 0]

    hsv_image = cv2.cvtColor(seg_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    mask_hsv = (h > 0) | (s > 0) | (v > 0)

    sum_hsv = np.sum(hsv_image, axis=2)
    sum_hsv = sum_hsv[sum_hsv != 0]

    return {
        'mean_b': np.mean(b[mask_rgb]),
        'mean_g': np.mean(g[mask_rgb]),
        'mean_r': np.mean(r[mask_rgb]),
        'max_rgb': np.max(sum_rgb),
        'min_rgb': np.min(sum_rgb),
        'std_b': np.std(b[mask_rgb]),
        'std_g': np.std(g[mask_rgb]),
        'std_r': np.std(r[mask_rgb]),
        'max_hsv': np.max(sum_hsv),
        'min_hsv': np.min(sum_hsv),
        'std_h': np.std(h[mask_hsv]),
        'std_s': np.std(s[mask_hsv]),
        'std_v': np.std(v[mask_hsv])
    }


def main(img: np.ndarray, mask: np.ndarray) -> str:
    """
    Uses segmented images and their characteristics to predict a category.
 
    Args:
        img (np.ndarray): The original image in BGR format.
        mask (np.ndarray): The binary mask for segmentation.

    Returns:
        str: "Один цвет" or "Больше одного цвета".
    """
    model = get_model()
    seg_img = segment_area_of_interest(img, mask)
    characteristics = extract_characteristics(seg_img)
    df = pd.DataFrame([characteristics])

    res = model.predict(df)
    return 'Один цвет' if res[0] == 1 else 'Больше одного цвета'
