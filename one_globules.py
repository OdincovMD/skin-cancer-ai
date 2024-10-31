import pandas as pd
from scipy import stats
import joblib
import cv2
import numpy as np

clf = joblib.load('weight/one_clods_single-color_clf.joblib')


def count_area_of_interest(img: np.ndarray) -> int:
    """
    Counts area of interest in image
    :param img: initial image
    :return: count of non-zero pixels in image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.countNonZero(gray)


def get_image_features(img: np.ndarray) -> dict:
    """
    Calculates features for image
    :param img: initial image
    :return: features dict
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


def classify_image(img: np.ndarray) -> str:
    """
    Classifies img
    :param img: image to classify
    :return: predicted label
    """
    features = get_image_features(img)
    df = pd.DataFrame([features])
    pred = clf.predict(df)

    return 'single_color' if pred[0] == 0 else 'several_colors'

def main(img: np.ndarray):
    label = classify_image(img)
    return label


if __name__ == '__main__':
    file_path = "26.jpg"  # change to your image path
    img = cv2.imread(file_path)
    result = main(img)
    print(result)
