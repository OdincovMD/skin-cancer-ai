import numpy as np
import cv2
import pandas as pd
from joblib import load
import math


gbc = load('weight/globules_one_multiple_color.joblib')


def extract_hsv_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    features = {
        'mean_h': np.mean(h),
        'mean_s': np.mean(s),
        'mean_v': np.mean(v),
        'std_h': np.std(h),
        'std_s': np.std(s),
        'std_v': np.std(v),
        'median_h': np.median(h),
        'median_s': np.median(s),
        'median_v': np.median(v),
        'max_h': np.max(h),
        'max_s': np.max(s),
        'max_v': np.max(v),
        'min_h': np.min(h),
        'min_s': np.min(s),
        'min_v': np.min(v)
    }
    return features

def color_deviation(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    return (np.std(h) + np.std(s) + np.std(v)) / 3.0

def average_color_rgb(img):
    avg_color = np.mean(img, axis=(0, 1)).astype(int)
    return list(reversed(avg_color))

def color_moments(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h_mean, s_mean, v_mean = np.mean(h), np.mean(s), np.mean(v)
    h_central_moment = np.mean((h - h_mean) ** 2)
    s_central_moment = np.mean((s - s_mean) ** 2)
    v_central_moment = np.mean((v - v_mean) ** 2)
    return h_mean, s_mean, v_mean, h_central_moment, s_central_moment, v_central_moment

def get_roundness(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else 0

def extract_blob_features(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    roundness = get_roundness(contour)
    return {'area': area, 'perimeter': perimeter, 'roundness': roundness}

def main(img):
    features = extract_hsv_features(img)
    df = pd.DataFrame([features])
    pred = gbc.predict(df)
    label = 'multiple_colors' if pred[0] == 1 else 'one_color'
    return label

if __name__ == "__main__":
    print(main(cv2.imread('26.jpg')))