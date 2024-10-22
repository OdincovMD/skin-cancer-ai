import base64

import pandas as pd
from scipy import stats
import cv2
import numpy as np
import joblib

CLF = joblib.load('weight/one_lines_branched_clf.joblib')

def count_area_of_interest(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.countNonZero(gray)

def get_image_features(img: np.ndarray) -> dict:
    features = {}
    area_value = count_area_of_interest(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)

    for channel, color in zip([b, g, r], ['b', 'g', 'r']):
        channel_nonzero = channel[channel != 0]
        features.update({
            f'mean_{color}': np.mean(channel_nonzero),
            f'mean_{color}/area_value': np.mean(channel_nonzero) / area_value,
            f'std_{color}': np.std(channel_nonzero),
            f'std_{color}/area_value': np.std(channel_nonzero) / area_value,
            f'var_{color}': np.var(channel_nonzero),
            f'var_{color}/area_value': np.var(channel_nonzero) / area_value,
            f'sum_{color}': np.sum(channel_nonzero),
            f'sum_{color}/area_value': np.sum(channel_nonzero) / area_value,
            f'max_{color}': np.max(channel_nonzero),
            f'max_{color}/area_value': np.max(channel_nonzero) / area_value,
            f'min_{color}': np.min(channel_nonzero),
            f'min_{color}/area_value': np.min(channel_nonzero) / area_value,
            f'median_{color}': np.median(channel_nonzero),
            f'median_{color}/area_value': np.median(channel_nonzero) / area_value,
            f'mode_{color}': float(stats.mode(channel_nonzero)[0]),
            f'mode_{color}/area_value': float(stats.mode(channel_nonzero)[0] / area_value)
        })

    features.update({
        'var_area_interest': np.var(gray),
        'std/area_value': np.std(gray) / area_value,
        'std_area_interest': np.std(gray),
        'mean/area_value': np.mean(gray) / area_value,
        'mean_area_interest': np.mean(gray),
        'var_area_interest/area_value': np.var(gray) / area_value,
        'area_value': area_value
    })

    return features


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = cv2.medianBlur(image, 3)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11,
                                               C=3)
    adaptive_threshold_inv = cv2.bitwise_not(adaptive_threshold)

    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    low = np.array([0, 0, 0])
    high = np.array([225, 125, 125])
    color_mask = cv2.inRange(segmented_image, low, high)
    segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=color_mask)
    segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=adaptive_threshold_inv)

    return segmented_image

def classify_image(img: np.ndarray) -> str:
    features = get_image_features(img)
    df = pd.DataFrame([features])
    pred = CLF.predict(df)
    return 'brown' if pred[0] == 0 else 'black'

def main(img: np.ndarray, mask: np.ndarray):
    segmented_img = apply_mask(img, mask)
    label = classify_image(segmented_img)
    return label

# if __name__ == '__main__':
#     file_path = "26.jpg"
#     img = cv2.imread(file_path)

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

#     result = main(img, mask)
#     print(result)