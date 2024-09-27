import cv2
import numpy as np
import pickle
import pandas as pd
from typing import Tuple

with open('weight/number_of_sign_lines_full.pkl', 'rb') as file:
    PIPELINE = pickle.load(file)

LABEL_MAPPING = {
    0: 'Curved',
    1: 'Parallel',
    2: 'Radial',
    3: 'Reticular_or_network'
}


def skeletonize(image: np.ndarray) -> np.ndarray:
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    ret, img = cv2.threshold(image, 128, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        done = zeros == size

    return skel


def detect_lines(img: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    return lines


def calculate_angles(lines: np.ndarray) -> np.ndarray:
    angles = []
    if lines is not None:
        angles = [np.degrees(line[0][1]) for line in lines]
    return np.array(angles)


def calculate_curvature(contour: np.ndarray) -> float:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    curvature = cv2.arcLength(approx, True) - cv2.arcLength(contour, True)
    return curvature


def get_skeleton_features(gray_img: np.ndarray) -> dict:
    skeletonized = skeletonize(gray_img)
    lines = detect_lines(skeletonized)
    angles = calculate_angles(lines)
    contours, _ = cv2.findContours(skeletonized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    branch_lengths = []
    curvature_values = []
    branch_count = 0
    for contour in contours:
        arc_length = cv2.arcLength(contour, True)
        if arc_length > 10:
            branch_lengths.append(arc_length)
            branch_count += 1
            curvature_values.append(calculate_curvature(contour))

    avg_branch_length = np.mean(branch_lengths) if branch_lengths else 0.0
    avg_curvature = np.mean(curvature_values) if curvature_values else 0.0
    avg_angle = np.mean(angles) if len(angles) else 0.0
    min_angle = np.min(angles) if len(angles) else 0.0
    max_angle = np.max(angles) if len(angles) else 0.0

    return {
        'branch_count': branch_count,
        'branch_lengths': avg_branch_length,
        'avg_curvature': avg_curvature,
        'avg_angle': avg_angle,
        'min_angle': min_angle,
        'max_angle': max_angle
    }


def blurring_gaussian(img: np.ndarray, k_size: Tuple[int, int] = (5, 5), sigma_x: int = 0,
                      sigma_y: int = 0) -> np.ndarray:
    return cv2.GaussianBlur(img, k_size, sigmaX=sigma_x, sigmaY=sigma_y)


def blurring_median(img: np.ndarray, k_size: int = 7) -> np.ndarray:
    return cv2.medianBlur(img, k_size)


def preprocessing_img(img: np.ndarray) -> np.ndarray:
    blurred = blurring_gaussian(blurring_median(img))
    return cv2.resize(blurred, (500, 500))


def region_of_lines_and_stuff(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)
    return cv2.bitwise_and(img, img, mask=mask)


def count_area_of_interest(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.countNonZero(thresh)


def get_image_features(img: np.ndarray) -> dict:
    region = region_of_lines_and_stuff(img)
    b, g, r = cv2.split(region)

    brown_mask = (r > 50) & (g > 30) & (b < 30)
    brown_pixel_count = np.sum(brown_mask)
    brown_area_ratio = brown_pixel_count / region.size

    features = {
        'std_b': np.std(b),
        'std_g': np.std(g),
        'std_r': np.std(r),
        'brown_area_ratio': brown_area_ratio,
        'area': count_area_of_interest(img)
    }
    features.update(get_skeleton_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    return features


def process_image_and_predict(image: np.ndarray) -> str:
    img = preprocessing_img(image)
    features = get_image_features(img)
    df = pd.DataFrame([features])
    predicted_label_encoded = PIPELINE.predict(df)[0]
    predicted_label = LABEL_MAPPING[predicted_label_encoded]
    return predicted_label


def main(image: np.ndarray) -> str:
    return process_image_and_predict(image)


if __name__ == "__main__":
    image_path = '26.jpg'
    image = cv2.imread(image_path)
    classification = main(image)
    print("Predicted class:", classification)
