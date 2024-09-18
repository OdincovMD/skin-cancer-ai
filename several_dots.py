import os
import cv2
import pandas as pd
import numpy as np
from joblib import load

CLF = load('weight/several_dots.joblib')

def calc_area_of_interest(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)
    return cv2.bitwise_and(img, img, mask=mask)

def get_image_features(img):
    area_interest = calc_area_of_interest(img)
    b, g, r = cv2.split(area_interest)
    return {
        'std_b': np.std(b),
        'std_g': np.std(g),
        'std_r': np.std(r)
    }

def classify_image(img):
    features = get_image_features(img)
    df = pd.DataFrame([features])
    pred = CLF.predict(df)
    return 'Black' if pred[0] == 1 else 'Brown'

def main(image):
    label = classify_image(image)
    return label

if __name__ == '__main__':
    image = cv2.imread('26.jpg')
    print(main(image))