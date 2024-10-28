import os
import pickle
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd

# TODO: добавить признаки и обучить модель again функции закоменчены в конце
# dictionary = {
#     'std_L': 'стандартное отклонение яркости',
#     'std_A': 'стандартное отклонение зеленого-красного',
#     'std_B': 'стандартное отклонение голубого-желтого',
#     'brown_area_ratio': 'отношение площади коричневых пикселей к общей площади',
#     'area': 'площадь области интереса',
#     'branch_count': 'количество ветвей',
#     'branch_lengths': 'средняя длина ветвей',
#     'avg_curvature': 'средняя кривизна контуров',
#     'contrast_haralick': 'контраст по Харалику',
#     'dissimilarity_haralick': 'несходство по Харалику',
#     'homogeneity_haralick': 'однородность по Харалику',
#     'energy_haralick': 'энергия по Харалику',
#     'correlation_haralick': 'корреляция по Харалику',
#     'mean_lbp': 'среднее значение локального бинарного паттерна',
#     'std_lbp': 'стандартное отклонение локального бинарного паттерна',
#     'median_lbp': 'медианное значение локального бинарного паттерна'
# }

#переименовано с number_of_sign_lines_full.pkl -> several_lines.pkl
MODEL_PATH = os.path.join('weight', 'several_lines.pkl')


def load_model(path: str = MODEL_PATH) -> Any:
    with open(path, 'rb') as file:
        return pickle.load(file)


_model_several_lines = None


def get_model() -> Any:
    """
    Retrieve the model, loading it from the file if it has not been loaded yet.

    Returns:
        Any: The loaded model object.
    """
    global _model_several_lines
    if not _model_several_lines:
        _model_several_lines = load_model()
    return _model_several_lines

LABEL_MAPPING = {
    0: 'Curved',
    1: 'Parallel',
    2: 'Radial',
    3: 'Reticular_or_network'
}

def get_skeleton_features(gray_img: np.ndarray) -> dict:
    """
    Extracts skeleton features from a grayscale image.

    Args:
        gray_img (np.ndarray): The grayscale image.

    Returns:
        dict: A dictionary containing counts of branches, average of branch lengths, curvatures and angles.
    """
    size = np.size(gray_img)
    skel = np.zeros(gray_img.shape, np.uint8)
    ret, img = cv2.threshold(gray_img, 128, 255, 0)
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

    edges = cv2.Canny(skel, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    angles = []
    if lines is not None:
        angles = [np.degrees(line[0][1]) for line in lines]

    contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    branch_lengths = []
    curvature_values = []
    branch_count = 0
    for contour in contours:
        arc_length = cv2.arcLength(contour, True)
        if arc_length > 10:
            branch_lengths.append(arc_length)
            branch_count += 1
            epsilon = 0.02 * arc_length
            approx = cv2.approxPolyDP(contour, epsilon, True)
            curvature = cv2.arcLength(approx, True) - arc_length
            curvature_values.append(curvature)

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


def process_image_and_extract_features(img: np.ndarray,
                                       apply_blur: bool = True,
                                       k_size_gaussian: Tuple[int, int] = (5, 5),
                                       sigma_x: int = 0, sigma_y: int = 0,
                                       k_size_median: int = 7) -> Dict[str, Any]:
    """
    Processes the input image by applying blurring, resizing, using a given mask and calculating features.

    Args:
        img (np.ndarray): The original image to process.
        mask (np.ndarray): Binary mask to isolate regions of interest.
        apply_blur (bool): Whether to apply blurring steps or not. Defaults to True.
        k_size_gaussian (Tuple[int, int]): The kernel size for the Gaussian blur. Defaults to (5, 5).
        sigma_x (int): Gaussian kernel standard deviation in the x-direction. Defaults to 0.
        sigma_y (int): Gaussian kernel standard deviation in the y-direction. Defaults to 0.
        k_size_median (int): Aperture linear size for median blur; it must be odd. Defaults to 7.

    Returns:
        dict: A dictionary containing various image feature statistics.
    """
    if apply_blur:
        img = cv2.medianBlur(img, k_size_median)
        img = cv2.GaussianBlur(img, k_size_gaussian, sigmaX=sigma_x, sigmaY=sigma_y)

    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = extract_image_features(img, gray)

    return features


def extract_image_features(region: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
    """
    Extracts various features from the image derived from the processed regions and skeleton features.

    Args:
        region (np.ndarray): Image with specific regions highlighted.
        gray (np.ndarray): Grayscale version of the original image.

    Returns:
        dict: A dictionary with various image feature statistics.
    """
    b, g, r = cv2.split(region)

    brown_mask = (r > 50) & (g > 30) & (b < 30)
    brown_pixel_count = np.sum(brown_mask)
    brown_area_ratio = brown_pixel_count / region.size

    features = {
        'std_b': np.std(b),
        'std_g': np.std(g),
        'std_r': np.std(r),
        'brown_area_ratio': brown_area_ratio,
        'area': cv2.countNonZero(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    }

    skeleton_features = get_skeleton_features(gray)
    features.update(skeleton_features)

    return features


def main(image: np.ndarray, mask: np.ndarray) -> str:
    """
    Main function to process and classify an image based on its features.

    Args:
        image (np.ndarray): The input image.

    Returns:
        str: "Curved", "Parallel", "Radial", "Reticular_or_network"
    """
    model = get_model()
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    features = process_image_and_extract_features(masked_image, apply_blur=True)
    df = pd.DataFrame([features])
    predicted_label_encoded = model.predict(df)[0]
    predicted_label = LABEL_MAPPING[predicted_label_encoded]
    return predicted_label


if __name__ == "__main__":
    image_path = '26.jpg'
    mask = None
    image = cv2.imread(image_path)
    classification = main(image, mask)
    print("Predicted class:", classification)

# def compute_haralick_features(image, mask):
#     """
#     Computes Haralick features
#     """
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     masked_gray = gray_image * (mask // 255)
#     masked_gray = masked_gray[mask == 255]

#     if masked_gray.size == 0:
#         print("Warning: Masked image has no elements!")
#         return {'contrast': 0.0, 'dissimilarity': 0.0, 'homogeneity': 0.0, 'energy': 0.0, 'correlation': 0.0}
#     distances = [1, 2, 3]
#     angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
#     glcm = graycomatrix(masked_gray.reshape(-1, 1), distances, angles, 256, symmetric=True, normed=True)

#     features = {
#         'contrast_haralick': np.mean(graycoprops(glcm, 'contrast')),
#         'dissimilarity_haralick': np.mean(graycoprops(glcm, 'dissimilarity')),
#         'homogeneity_haralick': np.mean(graycoprops(glcm, 'homogeneity')),
#         'energy_haralick': np.mean(graycoprops(glcm, 'energy')),
#         'correlation_haralick': np.mean(graycoprops(glcm, 'correlation'))
#     }
#     return features


# def compute_lbp(image, P=8, R=1):
#     """
#     Computes Local Binary Pattern (LBP)
#     """
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lbp = local_binary_pattern(gray_image, P, R, method='uniform')
#     return lbp


# def compute_lbp_statistics(lbp):
#     """
#     Computes statistics based on Local Binary Pattern (LBP)
#     """
#     return {
#         'mean_lbp': np.mean(lbp),
#         'std_lbp': np.std(lbp),
#         'min_lbp': np.min(lbp),
#         'max_lbp': np.max(lbp),
#         'median_lbp': np.median(lbp)
#     }

