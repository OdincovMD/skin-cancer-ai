import cv2
import numpy as np
import pickle
import pandas as pd
from typing import Tuple

# TODO: добавить признаки и обучить модель again функции закоменчены в конце
#dictionary = {
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


with open('weight/number_of_sign_lines_full.pkl', 'rb') as file:
    PIPELINE = pickle.load(file)

LABEL_MAPPING = {
    0: 'Curved',
    1: 'Parallel',
    2: 'Radial',
    3: 'Reticular_or_network'
}


def skeletonize(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to its skeleton representation using morphological transformations.

    Args:
        image (np.ndarray): The original binary image.

    Returns:
        np.ndarray: The skeletonized version of the input image.
    """
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
    """
    Detects lines in an image using the Canny edge detector and Hough line transform.

    Args:
        img (np.ndarray): The input image after preprocessing.

    Returns:
        np.ndarray: Array of detected lines. Each line is represented by rho and theta values.
    """
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    return lines


def calculate_angles(lines: np.ndarray) -> np.ndarray:
    """
    Calculate the angles of the detected lines.

    Args:
        lines (np.ndarray): Array of lines detected, each described by rho and theta.

    Returns:
        np.ndarray: Array of angles in degrees.
    """
    angles = []
    if lines is not None:
        angles = [np.degrees(line[0][1]) for line in lines]
    return np.array(angles)


def calculate_curvature(contour: np.ndarray) -> float:
    """
    Calculates the curvature for a given contour.

    Args:
        contour (np.ndarray): The contour of which to calculate the curvature.

    Returns:
        float: The calculated curvature value.
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    curvature = cv2.arcLength(approx, True) - cv2.arcLength(contour, True)
    return curvature


def get_skeleton_features(gray_img: np.ndarray) -> dict:
    """
    Extracts various features from the skeleton of an image.

    Args:
        gray_img (np.ndarray): Grayscale version of the image.

    Returns:
        dict: A dictionary containing counts of branches, average of branch lengths, curvatures and angles.
    """
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
    """
    Applies Gaussian blurring to an image which reduces image noise and detail.

    Args:
        img (np.ndarray): The input image to blur.
        k_size (Tuple[int, int]): The kernel size for the Gaussian operator.
        sigma_x (int): Gaussian kernel standard deviation in the x-direction.
        sigma_y (int): Gaussian kernel standard deviation in the y-direction.

    Returns:
        np.ndarray: The blurred image.
    """
    return cv2.GaussianBlur(img, k_size, sigmaX=sigma_x, sigmaY=sigma_y)


def blurring_median(img: np.ndarray, k_size: int = 7) -> np.ndarray:
    """
    Applies median blurring to the image, which is effective in removing salt-and-pepper noise.

    Args:
        img (np.ndarray): The input image.
        k_size (int): Aperture linear size; it must be odd and greater than 1.

    Returns:
        np.ndarray: The median blurred image.
    """
    return cv2.medianBlur(img, k_size)


def preprocessing_img(img: np.ndarray) -> np.ndarray:
    """
    Processes an input image by applying median and Gaussian blurring, followed by resizing.

    Args:
        img (np.ndarray): The original image to process.

    Returns:
        np.ndarray: The processed image resized to (500, 500) pixels.
    """
    blurred = blurring_gaussian(blurring_median(img))
    return cv2.resize(blurred, (500, 500))


def region_of_lines_and_stuff(img: np.ndarray) -> np.ndarray:
    """
    Identifies and isolates specific regions of interest in an image by applying a threshold and creating a mask.

    Args:
        img (np.ndarray): The original image.

    Returns:
        np.ndarray: Image with enhanced features using the mask.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)
    return cv2.bitwise_and(img, img, mask=mask)


def count_area_of_interest(img: np.ndarray) -> int:
    """
    Counts the area of interest in the image by thresholding and counting non-zero pixels.

    Args:
        img (np.ndarray): The original image.

    Returns:
        int: The count of non-zero pixels in the thresholded area.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.countNonZero(thresh)


def get_image_features(img: np.ndarray) -> dict:
    """
    Extracts various features from the image to be used for further analysis or model prediction.

    Args:
        img (np.ndarray): The preprocessed image.

    Returns:
        dict: A dictionary containing various image feature statistics including brown area ratio and standard deviations of RGB channels.
    """
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


def main(image: np.ndarray) -> str:
    """
    Main function to process and classify an image based on its features.

    Args:
        image (np.ndarray): The input image.

    Returns:
        str: The predicted label categorizing the image's line patterns.
    """
    img = preprocessing_img(image)
    features = get_image_features(img)
    df = pd.DataFrame([features])
    predicted_label_encoded = PIPELINE.predict(df)[0]
    predicted_label = LABEL_MAPPING[predicted_label_encoded]
    return predicted_label


if __name__ == "__main__":
    image_path = '26.jpg'
    image = cv2.imread(image_path)
    classification = main(image)
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