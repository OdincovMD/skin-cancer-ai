import pandas as pd
import cv2
import numpy as np
from scipy import stats
import joblib

clf = joblib.load('weight/one_globules.joblib')

def count_area_of_interest(img: np.ndarray) -> int:
    """
    Counts the number of pixels in the area of interest of the image.

    :param img: input image (three-channel)
    :return: number of non-zero pixels in the image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.countNonZero(gray)

def get_image_features(img: np.ndarray) -> dict:
    """
    Computes image features for classification.

    :param img: input image (three-channel)
    :return: dictionary of image features
    """
    features = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    area_value = count_area_of_interest(img)
    b, g, r = cv2.split(img)

    for channel, color in zip([b, g, r], ['b', 'g', 'r']):
        channel_nonzero = channel[channel != 0]
        if len(channel_nonzero) == 0:
            channel_nonzero = np.array([0])
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
            f'mode_{color}': float(stats.mode(channel_nonzero, keepdims=False)[0]),
            f'mode_{color}/area_value': float(stats.mode(channel_nonzero, keepdims=False)[0] / area_value)
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

def classify_image(img: np.ndarray) -> str:
    """
    Classifies the image based on a pre-trained model.

    :param img: image to classify
    :return: predicted label ('Один цвет' or 'Более одного цвета')
    """
    features = get_image_features(img)
    df = pd.DataFrame([features])
    pred = clf.predict(df)
    return 'Один цвет' if pred[0] == 0 else 'Более одного цвета'

def main(img: np.ndarray) -> str:
    """
    Main function for image classification.

    :param img: image to classify
    :return: 'Один цвет', 'Более одного цвета'
    """
    return classify_image(img)