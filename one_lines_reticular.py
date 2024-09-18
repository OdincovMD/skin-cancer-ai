import cv2
import numpy as np
import pickle
import pandas as pd

with open('weight/one_lines_reticular.pkl', 'rb') as file:
    PIPELINE = pickle.load(file)

LABEL_DICT = {0: '1_color', 1: 'more_colors'}


def extract_color_features(image):
    # Разделение на каналы
    b, g, r = cv2.split(image)

    # Исключаем значения равные 0
    nonzero_b = b[b != 0]
    nonzero_g = g[g != 0]
    nonzero_r = r[r != 0]

    return {
        "mean_blue": np.mean(nonzero_b),
        "median_blue": np.median(nonzero_b),
        "std_blue": np.std(nonzero_b),
        "mean_green": np.mean(nonzero_g),
        "median_green": np.median(nonzero_g),
        "std_green": np.std(nonzero_g),
        "mean_red": np.mean(nonzero_r),
        "median_red": np.median(nonzero_r),
        "std_red": np.std(nonzero_r)
    }


def extract_gray_features(image_gray):
    # Исключаем значения равные 0
    nonzero_gray = image_gray[image_gray != 0]

    return {
        "mean_gray": np.mean(nonzero_gray),
        "median_gray": np.median(nonzero_gray),
        "std_gray": np.std(nonzero_gray)
    }


def extract_texture_features(image_gray):
    moments = cv2.moments(image_gray)

    return {
        "energy": moments['nu20'] + moments['nu02'],
        "contrast": moments['nu20'] - moments['nu02']
    }


def region_of_lines_and_stuff(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    main_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined_thresh = cv2.bitwise_or(main_threshold, thresh)

    mask = cv2.bitwise_not(combined_thresh)
    return cv2.bitwise_and(img, img, mask=mask)


def extract_image_features(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = region_of_lines_and_stuff(cv2.resize(img, dsize=(750, 750)))

    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return {
        **extract_color_features(img),
        **extract_gray_features(image_gray),
        **extract_texture_features(image_gray)
    }


def process_image_and_predict(img):
    image_features = extract_image_features(img)
    df = pd.DataFrame([image_features])

    predicted_labels_encoded = PIPELINE.predict(df)
    predicted_label = LABEL_DICT[predicted_labels_encoded[0]]
    return predicted_label


def main(image_path):
    image = cv2.imread(image_path)
    return process_image_and_predict(image)



if __name__ == "__main__":
    image_path = "26.jpg"
    result = main(image_path)
    print(result)