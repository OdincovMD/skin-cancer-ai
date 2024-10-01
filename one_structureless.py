import os
import numpy as np
import cv2
import torch
import torchvision

class StructurelessClf_1(torch.nn.Module):
    def __init__(self, model_params={'in_size': 250000, 'layers': (100, 10), 'out_size': 1}):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['out_size'])

    def forward(self, X):
        X = self.conv1(X)
        X = torch.nn.functional.relu(X)
        X = torch.nn.functional.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = torch.nn.functional.relu(X)

        X = X.view(-1, 250000)

        X = self.fc1(X)
        X = torch.nn.functional.relu(X)

        X = self.fc2(X)
        X = torch.nn.functional.relu(X)
        X = self.fc3(X)
        X = X.squeeze()

        return X

MODEL = StructurelessClf_1()
MODEL.load_state_dict(torch.load('weight/One_Structureless_MonoMulti.pt', map_location=torch.device('cpu')))
DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL = MODEL.to(DEVICE)
TRANSFORM = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_tumor_contour(img: np.ndarray, solve_hair: bool = True, area_ratio_thresh: float = 0.65,
                      indent_ratio_thresh: float = 0.18) -> np.ndarray:
    """
    Получает контур опухоли на изображении.

    :param img: исходное изображение
    :param solve_hair: решать ли проблему с волосами (ухудшает качество сегментации, но игнорирует волосы)
    :param area_ratio_thresh: максимальное отношение площади подозрительной области к общей площади объекта,
                              чтобы считать его опухолью
    :param indent_ratio_thresh: максимальное отношение отступа по оси x (или y) к ширине (или высоте) объекта,
                                чтобы считать его опухолью
    :return: контур опухоли
    """
    height, width = img.shape[:2]
    area_thresh = area_ratio_thresh * height * width
    x_indent_left = indent_ratio_thresh * width
    x_indent_right = width - x_indent_left
    y_indent_upper = indent_ratio_thresh * height
    y_indent_lower = height - y_indent_upper

    best_contours = []
    orig = img

    for mode in range(3):
        if mode == 0:
            img_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        elif mode == 1:
            img_gray, _, _ = cv2.split(orig)
        elif mode == 2:
            img_b, img_g, _ = cv2.split(orig)
            img_gray = cv2.merge([img_b, img_g, img_b])
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

        k = get_k(img_gray, intensity_thresh=140)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * k
        img_filtered = cv2.filter2D(img_gray, -1, kernel)
        img_blurred = cv2.medianBlur(img_filtered, 15)

        _, mask = cv2.threshold(img_blurred, 110, 255, cv2.THRESH_BINARY_INV)
        if solve_hair:
            open_kernel = np.ones((9, 9))
            mask = cv2.erode(mask, open_kernel, iterations=5)
            mask = cv2.dilate(mask, open_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        best_contour = get_best_contour(contours, area_thresh, x_indent_left, x_indent_right,
                                        y_indent_upper, y_indent_lower)

        if best_contour is not None:
            best_contours.append(best_contour)

    if not best_contours:
        return None

    best_contour = get_best_contour_by_ratio(best_contours)
    best_contour = cv2.convexHull(best_contour)

    return best_contour


def get_k(img: np.ndarray, intensity_thresh: int = 145) -> float:
    """
    Вычисляет коэффициент для матрицы фильтра экспозиции, соответствующей заданной целевой интенсивности.

    :param img: исходное изображение
    :param intensity_thresh: целевая интенсивность результирующего изображения
    :return: коэффициент матрицы фильтра
    """
    def calc_intensity(img):
        return np.mean(img)

    k = 0.3
    while k < 2.2:
        k += 0.05
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * k
        intensity = calc_intensity(cv2.filter2D(img, -1, kernel))
        if intensity > intensity_thresh:
            return k

    return k


def get_best_contour(contours, area_thresh, x_indent_left, x_indent_right,
                     y_indent_upper, y_indent_lower):
    """
    Находит лучший контур среди заданных контуров.

    :param contours: список контуров
    :param area_thresh: максимальная площадь контура, чтобы считать его опухолью
    :param x_indent_left: левый отступ по оси x
    :param x_indent_right: правый отступ по оси x
    :param y_indent_upper: верхний отступ по оси y
    :param y_indent_lower: нижний отступ по оси y
    :return: лучший контур или None, если не найден
    """
    max_area = 0
    best_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= area_thresh and area > max_area:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if x_indent_left < cx < x_indent_right and y_indent_upper < cy < y_indent_lower:
                max_area = area
                best_contour = contour

    return best_contour


def get_best_contour_by_ratio(contours):
    """
    Находит лучший контур среди заданных контуров на основе отношения длины контура к площади.

    :param contours: список контуров
    :return: лучший контур
    """
    best_ratio = cv2.arcLength(contours[0], True) / np.sqrt(cv2.contourArea(contours[0]))
    best_contour = contours[0]

    for contour in contours:
        ratio = cv2.arcLength(contour, True) / np.sqrt(cv2.contourArea(contour))
        if ratio <= best_ratio:
            best_ratio = ratio
            best_contour = contour

    return best_contour


def get_cropped_image(img: np.ndarray, size_step: int = 224, const_res: bool = True, solve_hair: bool = True,
                      apply_mask: bool = False, draw_contour: bool = False,
                      contour_color: tuple[int, int, int] = (255, 0, 0),
                      area_ratio_thresh: float = 0.65, indent_ratio_thresh: float = 0.18) -> np.ndarray:
    """
    Возвращает обрезанное изображение с разрешением (N*size_step, N*size_step), где N - минимальное подходящее число.
    Опционально применяет маску к изображению и/или рисует контур на нем.

    :param img: исходное изображение
    :param size_step: шаг увеличения размера (равен минимальному из HEIGHT и WIDTH, если задан как 0 или меньше)
    :param const_res: изменять ли разрешение всех изображений до (size_step, size_step) после обработки
    :param solve_hair: решать ли проблему с волосами (ухудшает качество сегментации, но более вероятно игнорирует волосы)
    :param apply_mask: применять ли маску к изображению
    :param draw_contour: рисовать ли выбранный контур
    :param contour_color: цвет нарисованного контура
    :param area_ratio_thresh: максимальное отношение площади подозрительной области к общей площади объекта,
                              чтобы считать его опухолью
    :param indent_ratio_thresh: максимальное отношение отступа по оси x (или y) к ширине (или высоте) объекта,
                                чтобы считать его опухолью
    :return: обрезанное изображение минимального из доступных дискретных размеров
    """
    height, width = img.shape[:2]
    contour = get_tumor_contour(img, solve_hair=solve_hair, area_ratio_thresh=area_ratio_thresh,
                                indent_ratio_thresh=indent_ratio_thresh)

    if apply_mask and contour is not None:
        mask = np.zeros_like(img[:, :, 0], dtype='uint8')
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        img = cv2.bitwise_and(img, img, mask=mask)

    if draw_contour:
        cv2.drawContours(img, [contour], -1, color=contour_color, thickness=2)

    if size_step <= 0:
        size_step = min(width, height)

    x, y, w, h = cv2.boundingRect(contour)
    size = get_cropped_size(w, h, size_step, height, width)

    x1, y1, x2, y2 = get_crop_coordinates(x, y, w, h, size, height, width)

    cropped_img = img[y1:y2, x1:x2]
    if const_res:
        cropped_img = cv2.resize(cropped_img, (size_step, size_step))

    return cropped_img


def get_cropped_size(w, h, size_step, height, width):
    """
    Вычисляет размер обрезанного изображения.

    :param w: ширина ограничивающего прямоугольника контура
    :param h: высота ограничивающего прямоугольника контура
    :param size_step: шаг увеличения размера
    :param height: высота исходного изображения
    :param width: ширина исходного изображения
    :return: размер обрезанного изображения
    """
    size = size_step
    while w > size or h > size:
        size += size_step
        if size > height or size > width:
            size -= size_step
            w -= size_step
            break

    return size


def get_crop_coordinates(x, y, w, h, size, height, width):
    """
    Вычисляет координаты обрезки изображения.

    :param x: координата x ограничивающего прямоугольника контура
    :param y: координата y ограничивающего прямоугольника контура
    :param w: ширина ограничивающего прямоугольника контура
    :param h: высота ограничивающего прямоугольника контура
    :param size: размер обрезанного изображения
    :param height: высота исходного изображения
    :param width: ширина исходного изображения
    :return: координаты обрезки (x1, y1, x2, y2)
    """
    x_left = (size - w) // 2
    x_right = size - w - x_left
    y_upper = (size - h) // 2
    y_lower = size - h - y_upper
    y1 = y - y_upper
    y2 = y + h + y_lower
    x1 = x - x_left
    x2 = x + w + x_right

    if x1 < 0:
        shift = -x1
        x1 += shift
        x2 += shift
    elif x2 > width:
        shift = x2 - width
        x1 -= shift
        x2 -= shift

    if y1 < 0:
        shift = -y1
        y1 += shift
        y2 += shift
    elif y2 > height:
        shift = y2 - height
        y1 -= shift
        y2 -= shift

    return x1, y1, x2, y2


def main(img: np.ndarray) -> str:
    """
    Предсказывает класс: 'monochrome' или 'multicolor'.

    :param img: BGR изображение
    :return: класс
    """
    img = get_cropped_image(img)
    img = cv2.resize(img, dsize=(250, 250))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))

    img = torch.tensor(img, device=DEVICE, dtype=torch.float)
    img /= 255
    img = torch.unsqueeze(img, 0)
    img = TRANSFORM(img)

    with torch.no_grad():
        img = img * 2 - 1
        y = MODEL(img)
        y = y.to('cpu')
        y = torch.sigmoid(y)

        res = 'multicolor' if y > 0.5 else 'monochrome'

    return res


if __name__ == "__main__":
  file_path = '26.jpg'
  img = cv2.imread(file_path)
  result = main(img)
  print(result)
