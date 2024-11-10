import cv2
import pandas as pd
from joblib import load

classifier = load('weight/one_globules_several_colors.joblib')

def extract_color_features(img: cv2.Mat) -> dict:
    """
    Extracts color features from the image, including average values of BGR and HSV channels.

    :param img: input image (three-channel)
    :return: dictionary containing the color features of the image
    """
    features = {}
    bgr_means = cv2.mean(img)[:3]
    hsv_means = cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[:3]

    features['mean_b'], features['mean_g'], features['mean_r'] = bgr_means
    features['mean_h'], features['mean_s'], features['mean_v'] = hsv_means

    return features

def main(img: cv2.Mat) -> str:
    """
    Main function for classifying the image based on extracted color features.

    :param img: image to classify
    :return: label indicating the class of the image ('Меланин' or 'Другой пигмент')
    """
    features = extract_color_features(img)
    df = pd.DataFrame([features])
    label = 'Меланин' if classifier.predict(df)[0] == 1 else 'Другой пигмент'
    return label

