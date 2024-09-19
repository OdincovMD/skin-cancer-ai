import base64

import cv2
import numpy as np
import pandas as pd
from joblib import load
from roboflow import Roboflow

clf = load('weight/one_dots.joblib')


def blob_detector(min_thresh: int, thresh_step: int = None, max_thresh: int = None, min_area: int = None) -> cv2.SimpleBlobDetector:
    """
    Create a Blob Detector with specified parameters.
    :param min_thresh: minimal threshold
    :param thresh_step: step of threshold
    :param max_thresh: maximal threshold
    :param min_area: minimal value of dot's area
    :return: Resultant Blob Detector
    """
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_thresh
    params.thresholdStep = thresh_step if thresh_step is not None else params.thresholdStep
    params.maxThreshold = max_thresh if max_thresh is not None else params.maxThreshold
    params.filterByArea = min_area is not None
    params.minArea = min_area if min_area is not None else params.minArea
    return cv2.SimpleBlobDetector_create(params)


def apply_clahe(img, clip_limit, tile_grid_size):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.
    :param img: input image
    :param clip_limit: limit of the contrast
    :param tile_grid_size: the number of tiles in the row and column
    :return: result of CLAHE
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = cv2.split(lab)[0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(l_channel)


def convert_dots_to_mask(im_with_keypoints):
    """
    Convert white circles in the image to white rounds in the mask.
    :param im_with_keypoints: image with white circles
    :return: image with white rounds
    """
    contour_mask = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_mask = np.zeros_like(contour_mask, dtype=np.uint8)
    cv2.drawContours(res_mask, contours, -1, 255, thickness=cv2.FILLED)
    return res_mask


def extract_features_from_image(img, mask_of_lesion):
    """
    Extract features from the image.
    :param img: input image
    :param mask_of_lesion: mask of pigmented skin lesion
    :return: list of features (value of green, blue, red, and gray color) for each dot in the image
    """
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


def calculate_result_features(dots_features):
    """
    Calculate result features from the list of dot features.
    :param dots_features: list of features from dots
    :return: result features
    """
    g, b, r, m = zip(*dots_features)
    return [
        np.mean(g), np.mean(b), np.mean(r), np.mean(m),
        np.std(g), np.std(b), np.std(r), np.std(m)
    ]


def classify_image(img, mask):
    """
    Classify the image.
    :param img: image to classify
    :param mask: mask of pigmented skin lesion
    :return: predicted label
    """
    features = calculate_result_features(extract_features_from_image(img, mask))
    df = pd.DataFrame([features])
    pred = clf.predict(df)
    return 'brown' if pred[0] == 1 else 'gray'

    return None


def main(img: np.ndarray, mask: np.ndarray):
    """
    Print the label of the image by its path.
    :param img: image to classify
    :param mask: mask of pigmented skin lesion
    """

    label = classify_image(img, mask)
    return label


if __name__ == '__main__':
    image_path = "26.jpg"  # change to your image path
    img = cv2.imread(image_path)

    rf = Roboflow(api_key="GmJT3lC4NInRGZJ2iEit")
    project = rf.workspace("neo-dmsux").project("neo-v6wzn")
    model = project.version(2).model

    data = model.predict("26.jpg").json()
    width = data['predictions'][0]['image']['width']
    height = data['predictions'][0]['image']['height']

    encoded_mask = data['predictions'][0]['segmentation_mask']
    mask_bytes = base64.b64decode(encoded_mask)
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask_image = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask_image == 1, 255, mask_image)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

    print(main(img, mask))
