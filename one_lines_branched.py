import base64

import pandas as pd
from scipy import stats
import cv2
import numpy as np
import joblib

CLF = joblib.load('weight/one_lines_branched_clf.joblib')

def count_area_of_interest(img: np.ndarray) -> int:
    """
    Function for counting of area of neoplasm

    Parameters
    ____________
        img : np.ndarray
            Segmented image

    Returns
    ____________
        int
            Area of neoplasm
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.countNonZero(gray)

def get_image_features(img: np.ndarray) -> dict:
    """
    Function for acquiring values of features

    Parameters
    ____________
        img : np.ndarray
            Segmented image

    Returns
    ____________
        features : dict
            Dictionary of feature values
    """
    features = {}
    area_value = count_area_of_interest(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)

    for channel, color in zip([b, g, r], ['b', 'g', 'r']):
        channel_nonzero = channel[channel != 0]
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
            f'mode_{color}': float(stats.mode(channel_nonzero)[0]),
            f'mode_{color}/area_value': float(stats.mode(channel_nonzero)[0] / area_value)
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


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:

    """
    Function for masking the image

    Parameters
    ____________
        image : np.ndarray
            Original image of neoplasm
        mask : np.ndarray
            Binary image where white stands for area of neoplasm

    Returns
    ____________
        segmented_image : np.ndarray
            Image with selected area of interest
    """
    image = cv2.medianBlur(image, 3)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11,
                                               C=3)
    adaptive_threshold_inv = cv2.bitwise_not(adaptive_threshold)

    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    low = np.array([0, 0, 0])
    high = np.array([225, 125, 125])
    color_mask = cv2.inRange(segmented_image, low, high)
    segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=color_mask)
    segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=adaptive_threshold_inv)

    return segmented_image

def classify_image(img: np.ndarray) -> str:
    """
    Function for image classification

    Parameters
    ____________
        img : np.ndarray
            Segmented image

    Returns
    ____________
        str
            Resulting class
    """
    features = get_image_features(img)
    df = pd.DataFrame([features])
    pred = CLF.predict(df)
    return 'Коричневые' if pred[0] == 0 else 'Черные'

def main(img: np.ndarray, mask: np.ndarray):
    """
    Classification of branched lines by color:
    brown or black

    Parameters
    ____________
        image : np.ndarray
            Original image of neoplasm
        mask : np.ndarray
            Binary image where white stands for area of neoplasm

    Returns
    ____________
        label : str
            color of branched lines according to classificator
            Коричневые, Черные
    """
    segmented_img = apply_mask(img, mask)
    label = classify_image(segmented_img)
    return label
