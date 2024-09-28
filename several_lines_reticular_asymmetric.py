import base64
import numpy as np
import pandas as pd
import requests
import cv2
import pickle


with open('weight/several_lines_reticOrBranch_asymmetric.pkl', 'rb') as file:
    clf = pickle.load(file)


def segment_area_of_interest(img: np.ndarray, mask: np.ndarray):
    seg_img = cv2.bitwise_and(img, img, mask=mask)

    return cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)


def extract_characteristics(seg_img: np.ndarray):
    b, g, r = cv2.split(seg_img)
    mask_rgb = (b > 0) | (g > 0) | (r > 0)

    sum_rgb = np.sum(seg_img, axis=2)
    sum_rgb = sum_rgb[sum_rgb != 0]

    hsv_image = cv2.cvtColor(seg_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    mask_hsv = (h > 0) | (s > 0) | (v > 0)

    sum_hsv = np.sum(hsv_image, axis=2)
    sum_hsv = sum_hsv[sum_hsv != 0]

    return {
        'mean_b': np.mean(b[mask_rgb]),
        'mean_g': np.mean(g[mask_rgb]),
        'mean_r': np.mean(r[mask_rgb]),
        'max_rgb': np.max(sum_rgb),
        'min_rgb': np.min(sum_rgb),
        'std_b': np.std(b[mask_rgb]),
        'std_g': np.std(g[mask_rgb]),
        'std_r': np.std(r[mask_rgb]),
        'max_hsv': np.max(sum_hsv),
        'min_hsv': np.min(sum_hsv),
        'std_h': np.std(h[mask_hsv]),
        'std_s': np.std(s[mask_hsv]),
        'std_v': np.std(v[mask_hsv])
    }


def main(img: np.ndarray, mask: np.ndarray):
    seg_img = segment_area_of_interest(img, mask)
    characteristics = extract_characteristics(seg_img)
    df = pd.DataFrame([characteristics])

    res = clf.predict(df)
    return 'ОДИН ЦВЕТ' if res[0] == 1 else 'БОЛЬШЕ ОДНОГО ЦВЕТА'


# if __name__ == "__main__":
#     img_path = '26.jpg'
#     img = cv2.imread(img_path)

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
