import cv2
import numpy as np
from math import sin, cos
from joblib import load

clf = load('weight/one_globules_morethanonecolor_melanin.joblib')

def preprocess_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = cv2.split(hsv)[0]
    kernel = np.ones((17, 17), np.uint8)
    thresh = cv2.morphologyEx(cv2.threshold(h_channel, 20, 255, cv2.THRESH_BINARY_INV)[1], cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return mask

def extract_nevus(img, mask):
    img_BGRA = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    masked = cv2.bitwise_and(img_BGRA, img_BGRA, mask=mask)
    masked[mask < 2] = [0, 0, 0, 0]
    background = cv2.GaussianBlur(img, (65, 65), 0)
    background_BGRA = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    return cv2.add(background_BGRA, masked)

def find_globule_centers(img, mask):
    src = cv2.morphologyEx(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.MORPH_BLACKHAT, np.ones((50, 50), np.uint8))
    kernel = np.ones((7, 7), np.uint8)
    byte = cv2.morphologyEx(cv2.threshold(src, 20, 255, cv2.THRESH_BINARY)[1], cv2.MORPH_OPEN, kernel)
    byte = cv2.morphologyEx(byte, cv2.MORPH_CLOSE, kernel)
    canny = cv2.Canny(byte, 30, 150)
    img_cnt = cv2.dilate(canny, (1, 1), iterations=0)
    contours, _ = cv2.findContours(img_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    x_array, y_array = [], []
    for cnt in contours:
        ((x, y), _) = cv2.minEnclosingCircle(cnt)
        x_array.append(x)
        y_array.append(y)
    return x_array, y_array

def find_nevus_center_and_area(mask):
    square = cv2.countNonZero(mask)
    moments = cv2.moments(mask)
    if moments["m00"] > 0:
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
    else:
        x, y = mask.shape[1] // 2, mask.shape[0] // 2
    return x, y, square

def align_object(mask):
    moments = cv2.moments(mask)
    mu02 = moments['mu02']
    mu11 = moments['mu11']
    mu20 = moments['mu20']
    cov_matrix = np.array([[mu20, mu11], [mu11, mu02]])
    _, eigenvectors = np.linalg.eigh(cov_matrix)
    angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    return angle

def align_arrays(x_array, y_array, angle):
    cos_a, sin_a = cos(angle), sin(angle)
    al_array_x = [x / cos_a - y / sin_a for x, y in zip(x_array, y_array)]
    al_array_y = [y / cos_a + x / sin_a for x, y in zip(x_array, y_array)]
    return al_array_x, al_array_y

def color_ratio(img, center, axis, x_array, y_array):
    b, g, r = cv2.split(img)
    channels = [b, g, r]
    ratios = []
    for channel in channels:
        total_sum = sum(channel[int(y), int(x)] for x, y in zip(x_array, y_array))
        part_sum = sum(channel[int(y), int(x)] for x, y in zip(x_array, y_array) if (y > center if axis == 'y' else x > center))
        ratios.append(part_sum / total_sum if total_sum > 0 else 0)
    return ratios

def standard_deviation(x_0, y_0, x_array, y_array):
    lst_std_xpos_y = [y for x, y in zip(x_array, y_array) if x > x_0]
    lst_std_xneg_y = [y for x, y in zip(x_array, y_array) if x <= x_0]
    lst_std_xpos_x = [x for x, y in zip(x_array, y_array) if x > x_0]
    lst_std_xneg_x = [x for x, y in zip(x_array, y_array) if x <= x_0]
    lst_std_ypos_y = [y for x, y in zip(x_array, y_array) if y > y_0]
    lst_std_yneg_y = [y for x, y in zip(x_array, y_array) if y <= y_0]
    lst_std_ypos_x = [x for x, y in zip(x_array, y_array) if y > y_0]
    lst_std_yneg_x = [x for x, y in zip(x_array, y_array) if y <= y_0]
    s1 = np.std(lst_std_ypos_x) / np.std(lst_std_yneg_x) if np.std(lst_std_yneg_x) != 0 else 0.0
    s2 = np.std(lst_std_ypos_y) / np.std(lst_std_yneg_y) if np.std(lst_std_yneg_y) != 0 else 0.0
    s3 = np.std(lst_std_xpos_x) / np.std(lst_std_xneg_x) if np.std(lst_std_xneg_x) != 0 else 0.0
    s4 = np.std(lst_std_xpos_y) / np.std(lst_std_xneg_y) if np.std(lst_std_xneg_y) != 0 else 0.0
    return s1, s2, s3, s4

def get_image_features(img: np.ndarray) -> dict:
    mask = preprocess_image(img)
    nevus = extract_nevus(img, mask)
    x_array, y_array = find_globule_centers(nevus, mask)
    x_0, y_0, square = find_nevus_center_and_area(mask)
    angle = align_object(mask)
    al_x_array, al_y_array = align_arrays(x_array, y_array, angle)
    al_x_0, al_y_0 = align_arrays([x_0], [y_0], angle)
    al_x_0, al_y_0 = al_x_0[0], al_y_0[0]  # Extract the values from the lists
    features = {}
    features['average_x'] = np.mean(np.abs(np.array(al_x_array) - al_x_0))
    features['average_y'] = np.mean(np.abs(np.array(al_y_array) - al_y_0))
    features['red_x'], features['green_x'], features['blue_x'] = color_ratio(img, y_0, 'x', x_array, y_array)
    features['red_y'], features['green_y'], features['blue_y'] = color_ratio(img, x_0, 'y', x_array, y_array)
    features['std_1'], features['std_2'], features['std_3'], features['std_4'] = standard_deviation(al_x_0, al_y_0, al_x_array, al_y_array)
    return features

def main(img: np.ndarray):
    features = get_image_features(img)
    pred = clf.predict([list(features.values())])
    if pred == 0:
        return 'АСИММЕТРИЧНОЕ РАСПОЛОЖЕНИЕ ГЛОБУЛ'
    if pred == 1:
        return 'СИММЕТРИЧНОЕ РАСПОЛОЖЕНИЕ ГЛОБУЛ'

if __name__ == "__main__":
    image_path = '26.jpg'
    image = cv2.imread(image_path)
    label = main(image)
    print(label)