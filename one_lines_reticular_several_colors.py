import cv2
import numpy as np
import pandas as pd
import os
import pickle

with open('weight/one_lines_retic_moreThanOneColor_1.pkl', 'rb') as file:
    clf_1 = pickle.load(file)
with open('weight/one_lines_retic_moreThanOneColor_2.pkl', 'rb') as file:
    clf_2 = pickle.load(file)


def calc_area_of_interest(img: np.ndarray) -> tuple:
    """
    Function for selecting area of interest

    Parameters
    ____________
        img : np.ndarray
            Original image of neoplasm

    Returns
    ____________
        res_image : np.ndarray
            Segmented image with selected area of neoplasm
        largest_contour : np.ndarray
            coordinates of contour of area of interest
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_areas = [(cv2.contourArea(contour), contour) for contour in contours]
    contour_areas = sorted(contour_areas, key=lambda x: x[0], reverse=True)
    top_5_contours = [contour[1] for contour in contour_areas[:5]]

    min_sum = float('inf')
    largest_contour = None
    res_img = None

    for contour in top_5_contours:
        if cv2.contourArea(contour) > 50000:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            r, g, b = cv2.split(masked_img)
            mean_sum = np.mean(r) + np.mean(g) + np.mean(b)

            if mean_sum < min_sum:
                min_sum = mean_sum
                largest_contour = contour
                res_img = masked_img

    return res_img, largest_contour


def calc_in_out_area(img: np.ndarray, contour: np.ndarray) -> tuple:
    """
    Function for calculating area of neoplasm and background

    Parameters
    ____________
        img : np.ndarray
            Segmented image
        contour : np.ndarray
            Contour coordinates of neoplasm

    Returns
    ____________
        inside_area : np.ndarray
            Segmented neoplasm
        outside_area : np.ndarray
            Segmented background
    """
    height, width, _ = img.shape
    area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    radius = int(np.sqrt(area) / 4)

    mask1 = np.zeros_like(img)
    cv2.circle(mask1, (center_x, center_y), radius, (255, 255, 255), -1)
    inside_area = cv2.bitwise_and(img, mask1)

    mask2 = np.ones_like(img) * 255
    cv2.circle(mask2, (center_x, center_y), radius, (0, 0, 0), -1)
    outside_area = cv2.bitwise_and(img, mask2)

    return inside_area, outside_area


def split_masked_image_into_micro_sectors(masked_image: np.ndarray, contour: np.ndarray) -> tuple:
    """
    Function for splitting neoplasm into sector area

    Parameters
    ____________
        masked_image : np.ndarray
            Segmented image
        contour : np.ndarray
            Contour coordinates of neoplasm

    Returns
    ____________
        sectors : array
            array of parts of neoplasm
        cols : int
            number of taken columns
    """
    height, width = masked_image.shape[:2]
    area = cv2.contourArea(contour)
    rows = cols = int(50 - (area // 75000))
    sector_width = width // cols
    sector_height = height // rows

    sectors = []
    for i in range(rows):
        for j in range(cols):
            x = j * sector_width
            y = i * sector_height
            sector = masked_image[y:y + sector_height, x:x + sector_width]
            sectors.append(sector)

    return sectors, cols


def count_characteristics1(img: np.ndarray) -> dict:
    """
    Function for calculating values of features for findind mixture of colors

    Parameters
    ____________
        img : np.ndarray
            Original image of neoplasm

    Returns
    ____________
        characters : dict
            dictionary with calculated values of features
    """
    characters = {}
    area_of_interest, contour = calc_area_of_interest(img)

    b, g, r = cv2.split(area_of_interest)
    mask_rgb = (b > 0) | (g > 0) | (r > 0)
    characters['mean_b'] = np.mean(b[mask_rgb])
    characters['mean_g'] = np.mean(g[mask_rgb])
    characters['mean_r'] = np.mean(r[mask_rgb])
    characters['std_b'] = np.std(b[mask_rgb])
    characters['std_g'] = np.std(g[mask_rgb])
    characters['std_r'] = np.std(r[mask_rgb])

    hsv_image = cv2.cvtColor(area_of_interest, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    mask_hsv = (h > 0) | (s > 0) | (v > 0)
    characters['mean_h'] = np.mean(h[mask_hsv])
    characters['mean_s'] = np.mean(s[mask_hsv])
    characters['mean_v'] = np.mean(v[mask_hsv])
    characters['std_h'] = np.std(h[mask_hsv])
    characters['std_s'] = np.std(s[mask_hsv])
    characters['std_v'] = np.std(v[mask_hsv])

    sectors, cols = split_masked_image_into_micro_sectors(area_of_interest, contour)
    df_rgb = pd.DataFrame(columns=range(cols), index=range(cols))

    for idx, sector in enumerate(sectors):
        i, j = idx // cols, idx % cols
        b, g, r = cv2.split(sector)
        mask = (b > 0) & (g > 0) & (r > 0)
        mean_b = np.mean(b[mask])
        mean_g = np.mean(g[mask])
        mean_r = np.mean(r[mask])
        if not np.isnan(mean_b):
            df_rgb.iloc[i, j] = mean_b + mean_g + mean_r

    pd.set_option('future.no_silent_downcasting', True)
    df_rgb = df_rgb.fillna(0)

    diffs = []
    for row in range(1, cols - 1):
        for col in range(1, cols - 1):
            diffs.append(abs(df_rgb.iloc[row, col] - df_rgb.iloc[row - 1, col]) +
                         abs(df_rgb.iloc[row, col] - df_rgb.iloc[row, col - 1]) +
                         abs(df_rgb.iloc[row, col] - df_rgb.iloc[row + 1, col]) +
                         abs(df_rgb.iloc[row, col] - df_rgb.iloc[row, col + 1]))
    characters['mean_diff'] = sum(diffs) / len(diffs) if diffs else 0

    return characters


def count_characteristics2(img: np.ndarray) -> dict:
    """
    Function for calculating values of features for findind position of pigmentation

    Parameters
    ____________
        img : np.ndarray
            Original image of neoplasm

    Returns
    ____________
        characters : dict
            dictionary with calculated values of features
    """
    characters = {}
    area_of_interest, contour = calc_area_of_interest(img)
    area_inside, area_outside = calc_in_out_area(area_of_interest, contour)

    in_b, in_g, in_r = cv2.split(area_inside)
    in_mask = (in_b > 0) | (in_g > 0) | (in_r > 0)
    characters['in_mean_b'] = np.mean(in_b[in_mask])
    characters['in_mean_g'] = np.mean(in_g[in_mask])
    characters['in_mean_r'] = np.mean(in_r[in_mask])
    characters['in_std_b'] = np.std(in_b[in_mask])
    characters['in_std_g'] = np.std(in_g[in_mask])
    characters['in_std_r'] = np.std(in_r[in_mask])

    out_b, out_g, out_r = cv2.split(area_outside)
    out_mask = (out_b > 0) | (out_g > 0) | (out_r > 0)
    characters['out_mean_b'] = np.mean(out_b[out_mask])
    characters['out_mean_g'] = np.mean(out_g[out_mask])
    characters['out_mean_r'] = np.mean(out_r[out_mask])
    characters['out_std_b'] = np.std(out_b[out_mask])
    characters['out_std_g'] = np.std(out_g[out_mask])
    characters['out_std_r'] = np.std(out_r[out_mask])

    sectors, _ = split_masked_image_into_micro_sectors(area_outside, contour)
    max_colors = [1, 1, 1]
    min_colors = [255, 255, 255]

    for sector in sectors:
        b, g, r = cv2.split(sector)
        mask = (b > 0) & (g > 0) & (r > 0)
        mean_b = np.mean(b[mask])
        mean_g = np.mean(g[mask])
        mean_r = np.mean(r[mask])
        if not np.isnan(mean_b):
            if mean_b + mean_g + mean_r > sum(max_colors):
                max_colors = [mean_b, mean_g, mean_r]
            if mean_b + mean_g + mean_r < sum(min_colors):
                min_colors = [mean_b, mean_g, mean_r]

    characters['out_rgb_distance'] = np.linalg.norm(np.array(max_colors) - np.array(min_colors))

    return characters


def main(img: np.ndarray) -> str:
    """
    Classification of reticular lines with several colors by color type:
    Reticular, Spread, Parallel or Curved

    Parameters
    ____________
        img : np.ndarray
            Original image of neoplasm

    Returns
    ____________
        result : str
            Type of multicolored reticular lines according to classificator
            Пестрый и краповый, Центральная гиперпигментация, Периферическая гиперпигментация
    """
    info_img_1 = count_characteristics1(img)
    df_1 = pd.DataFrame(info_img_1, index=[0])
    res_1 = clf_1.predict(df_1)

    if res_1 == 0:
        return 'Пестрый и краповый'
    else:
        info_img_2 = count_characteristics2(img)
        df_2 = pd.DataFrame(info_img_2, index=[0])
        res_2 = clf_2.predict(df_2)
        if res_2 == 0:
            return 'Центральная гиперпигментация'
        else:
            return 'Периферическая гиперпигментация'


