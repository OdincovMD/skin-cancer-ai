import numpy as np
import cv2
from joblib import load

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the image by enhancing contrast and applying adaptive thresholding.

    Args:
        img (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: Thresholded binary image.
    """
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 20, 100)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 101, -13)
    return thresh


def detect_keypoints(img: np.ndarray) -> list:
    """
    Detect keypoints in the preprocessed binary image using blob detection.

    Args:
        img (np.ndarray): Preprocessed binary image.

    Returns:
        list: A list of detected keypoints.
    """
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = True
    params.minConvexity = 0.001
    params.filterByInertia = True
    params.minInertiaRatio = 0.001

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    return keypoints


def extract_features(img: np.ndarray, keypoints: list) -> list:
    """
    Extract color features from the area surrounding each keypoint.

    Args:
        img (np.ndarray): Original image from which features are extracted.
        keypoints (list): List of detected keypoints.

    Returns:
        list: List of extracted color features.
    """
    features = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        bgr_values = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if 0 <= y + dy < img.shape[0] and 0 <= x + dx < img.shape[1]:
                    bgr_values.append(img[y + dy, x + dx])

        bgr_values = np.array(bgr_values, dtype=np.int16)
        min_idx = np.argmin(np.sum(bgr_values, axis=1))
        b, g, r = bgr_values[min_idx]

        features.append([b, g, r])

    features = np.array(features)
    b_mean, g_mean, r_mean = np.mean(features, axis=0)
    b_min, g_min, r_min = np.min(features, axis=0)
    b_max, g_max, r_max = np.max(features, axis=0)

    return [b_mean, g_mean, r_mean, b_min, g_min, r_min, b_max, g_max, r_max]


def classify_image(features: list, clf) -> str:
    """
    Classify the image based on extracted features using a pre-trained classifier.

    Args:
        features (list): List of extracted color features.
        clf: Pre-trained classifier object.

    Returns:
        str: Classification result ("Коричневый" or "Черный или серый").
    """
    X = np.array(features).reshape(1, -1)
    y_pred = clf.predict(X)
    return "Коричневый" if y_pred[0] == 0 else "Черный или серый"

clf = load('weight/two_circles.joblib')


def main(img: np.ndarray, mask: np.ndarray) -> str:
    """
    Main function to process the image and classify it based on the mask and features.

    Args:
        img (np.ndarray): Original image.
        mask (np.ndarray): Binary mask defining the region of interest.

    Returns:
        str: Classification result ("Коричневый" or "Черный или серый").
    """
    masked = cv2.bitwise_and(img, img, mask=mask)
    preprocessed = preprocess_image(masked)
    keypoints = detect_keypoints(preprocessed)
    features = extract_features(masked, keypoints)
    result = classify_image(features, clf)
    return result


# if __name__ == "__main__":
#     img_path = '26.jpg'
#     img = cv2.imread(img_path)
#
#     rf = Roboflow(api_key="GmJT3lC4NInRGZJ2iEit")
#     project = rf.workspace("neo-dmsux").project("neo-v6wzn")
#     model = project.version(2).model
#
#     data = model.predict("26.jpg").json()
#     width = data['predictions'][0]['image']['width']
#     height = data['predictions'][0]['image']['height']
#
#     encoded_mask = data['predictions'][0]['segmentation_mask']
#     mask_bytes = base64.b64decode(encoded_mask)
#     mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
#     mask_image = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
#     mask = np.where(mask_image == 1, 255, mask_image)
#     mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
#
#     res = main(img, mask)
#     print(res)