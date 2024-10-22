import cv2
import pandas as pd
import numpy as np
from joblib import load

clf = load('weight/several_lines_radial_edges_pereferic.joblib')


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = np.mean(gray[0, :]) + np.mean(gray[-1, :]) + np.mean(gray[:, 0]) + np.mean(gray[:, -1])
    bg /= (2 * (gray.shape[0] + gray.shape[1]))
    mask = cv2.inRange(gray, bg - 45, bg + 45)
    return mask


def get_image_features(img):
    mask = preprocess_image(img)
    r, g, b = cv2.mean(img, mask=mask)[:3]
    return {'r': r, 'g': g, 'b': b}


def main(img):
    features = get_image_features(img)
    df = pd.DataFrame([features])
    pred = clf.predict(df)
    return 'СВЕТЛАЯ БЕССТРУКТУРНАЯ ОБЛАСТЬ' if pred == 1 else 'ТЕМНАЯ БЕССТРУКТУРНАЯ ОБЛАСТЬ'


if __name__ == "__main__":
    image_path = "26.jpg"
    image = cv2.imread(image_path)
    label = main(image)
    print(label)
