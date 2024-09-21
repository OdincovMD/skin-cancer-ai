import base64

import cv2
import numpy as np
import pandas as pd
from joblib import load
from roboflow import Roboflow
from typing import Optional

clf = load('./weight/one_dots.joblib')

def blob_detector(min_thresh: int,
                  thresh_step: Optional[int] = None,
                  max_thresh: Optional[int] = None,
                  min_area: Optional[int] = None) -> cv2.SimpleBlobDetector:
    '''
    Create a Blob Detector with specified parameters.

    Parameters
    ----------
        min_thresh : int
            The lower bound of threshold.
        thresh_step : Optional[int]
            Step of threshold.
        max_thresh : Optional[int]
            The upper bound of threshold.
        min_area : Optional[int]
            Minimum value of dot's area.

    Returns
    -------
        cv2.SimpleBlobDetector
            Resultant Blob Detector.
    '''
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_thresh
    if thresh_step:
        params.thresholdStep = thresh_step
    if max_thresh:
        params.maxThreshold = max_thresh
    if min_area:
        params.minArea = min_area
    params.filterByArea = bool(min_area)
    return cv2.SimpleBlobDetector_create(params)


def apply_clahe(img: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
    '''
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.

    Parameters
    ----------
        img : np.ndarray
            Input image.
        clip_limit : float
            Limit of the contrast.
        tile_grid_size : tuple[int, int]
            The number of tiles in the row and column.

    Returns
    -------
        np.ndarray
            Image with applied CLAHE.
    '''
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = cv2.split(lab)[0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(l_channel)


def convert_dots_to_mask(im_with_keypoints: np.ndarray) -> np.ndarray:
    '''
    Convert white circles in the image to white rounds in the mask.

    Parameters
    ----------
        im_with_keypoints : np.ndarray
            Iimage with white circles.

    Returns
    -------
        np.ndarray
            Mask that contains white rounds.    
    '''
    contour_mask = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_mask = np.zeros_like(contour_mask, dtype=np.uint8)
    cv2.drawContours(res_mask, contours, -1, 255, thickness=cv2.FILLED)
    return res_mask


def extract_features_from_image(img: np.ndarray, mask_of_lesion: np.ndarray) -> list[int]:
    '''
    Extract features from the image.

    Parameters
    ----------
        im_with_keypoints : np.ndarray
            Image with white circles.
        mask_of_lesion : np.ndarray
            Mask of pigmented skin lesion.

    Returns
    -------
        list[int]
            List of features (value of green, blue, red, and gray color) for each dot in the image.
    '''
    dots_features = []
    keypoints = []
    limit = 0.1
    im_with_keypoints = np.zeros_like(img, dtype=np.uint8)

    while len(keypoints) < 10:
        limit += 0.5
        equalized = apply_clahe(img, limit, (6, 4))
        equalized = cv2.bilateralFilter(equalized, 12, 75, 75)
        detector = blob_detector(0, 1, 40, 20)
        keypoints = detector.detect(equalized, mask=mask_of_lesion)
        im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, keypoints, np.array([]), (255, 255, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    res_mask = convert_dots_to_mask(im_with_keypoints)
    contours, _ = cv2.findContours(res_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        contour_mask = np.zeros_like(res_mask, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        average_color = cv2.mean(img, mask=contour_mask)[:3]
        dots_features.append(list(average_color) + [sum(average_color) / len(average_color)])

    return dots_features


def calculate_result_features(dots_features: list[int]) -> list[int]:
    '''
    Calculate result features from the list of dot features.

    Parameters
    ----------
        dots_features : list[int]
            List of features from dots.

    Returns
    -------
        list[int]
            Result features.
    '''
    g, b, r, m = zip(*dots_features)
    return [
        np.mean(g), np.mean(b), np.mean(r), np.mean(m),
        np.std(g), np.std(b), np.std(r), np.std(m)
    ]


def classify_image(img: np.ndarray, mask: np.ndarray) -> str:
    '''
    Classify the image.

    Parameters
    ----------
        img : np.ndarray
            The original image of the neoplasm.
        mask : np.ndarray
            Mask of pigmented skin lesion.

    Returns
    -------
        str
            "Коричневый"; "Серый".
    '''
    features = calculate_result_features(extract_features_from_image(img, mask))
    df = pd.DataFrame([features])
    pred = clf.predict(df)
    return 'brown' if pred[0] == 1 else 'gray'


def main(img: np.ndarray, mask: np.ndarray) -> str:
    '''
    Classification of a neoplasm by color within an area that contains dots.

    Parameters
    ----------
        img : np.ndarray
            The original image of the neoplasm.
        mask : np.ndarray
            Mask of pigmented skin lesion.

    Returns
    -------
        str
            "Коричневый"; "Серый".
    '''
    label = classify_image(img, mask)
    return label


# Deprecated since the mask is being passed from the main.py
# if __name__ == '__main__':
#     image_path = "26.jpg"  # change to your image path
#     img = cv2.imread(image_path)

#     rf = Roboflow(api_key="GmJT3lC4NInRGZJ2iEit")
#     project = rf.workspace("neo-dmsux").project("neo-v6wzn")
#     model = project.version(2).model

#     data = model.predict("26.jpg").json()
#     width = data['predictions'][0]['image']['width']
#     height = data['predictions'][0]['image']['height']

#     encoded_mask = data['predictions'][0]['segmentation_mask']
#     mask_bytes = base64.b64decode(encoded_mask)
#     mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
#     mask_image = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
#     mask = np.where(mask_image == 1, 255, mask_image)
#     mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

#     print(main(img, mask))
