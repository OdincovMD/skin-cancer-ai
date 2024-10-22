import numpy as np
import cv2
import torch
import torchvision

import mask_builder # Есил это для локального тетса, то окей. Если в проекте нужна маска, то сообщить об этом.

class StructurelessClf_1(torch.nn.Module):
    """
    Сверточная нейронная сеть для определения класса изображения
    """
    def __init__(self, model_params={'in_size': 250000, 'layers': (100, 10), 'out_size': 1}):
        """
        Иницилизация структуры нейронной сети

        :param model_params: параметры модели
        #in_size: размерность входного вектора
        #layers: размерность скрытых слоев
        #out_size: размер выходного вектора (количество классов)
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, 1, 1, bias=True)

        self.fc1 = torch.nn.Linear(model_params['in_size'], model_params['layers'][0])
        self.fc2 = torch.nn.Linear(model_params['layers'][0], model_params['layers'][1])
        self.fc3 = torch.nn.Linear(model_params['layers'][1], model_params['out_size'])

    def forward(self, X):
        """
        Проход по нейронной сети

        :param X: принимаемое изображение
        :return: класс объекта
        """
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
MODEL.load_state_dict(torch.load('weight/one_structureless.pt', map_location=torch.device('cpu')))
DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL = MODEL.to(DEVICE)
TRANSFORM = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_cropped_image(img: np.ndarray, mask: np.ndarray, size_step: int = 224, const_res: bool = True,
                      solve_hair: bool = True, apply_mask: bool = True,
                      contour_color: tuple[int, int, int] = (255, 0, 0),
                      area_ratio_thresh: float = 0.65, indent_ratio_thresh: float = 0.18) -> np.ndarray:
    """
    Возвращает обрезанное изображение с разрешением (N*size_step, N*size_step), где N - минимальное подходящее число.
    Опционально применяет маску к изображению и/или рисует контур на нем.

    :param img: исходное изображение
    :param mask: маска к изображению
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
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    contour = contours[0]

    if apply_mask and contour is not None:
        mask = np.zeros_like(img[:, :, 0], dtype='uint8')
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        img = cv2.bitwise_and(img, img, mask=mask)

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


def main(img: np.ndarray, mask: np.ndarray) -> str:
    """
    Предсказывает класс: 'monochrome' или 'multicolor'.

    :param img: BGR изображение
    :param mask: маска к изображению
    :return: класс
    """
    img = get_cropped_image(img, mask)
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

if __name__ == '__main__':
    path_to_image = '619.jpg'
    img = cv2.imread(path_to_image)
    mask = mask_builder.main(path_to_image)
    print(main(img, mask))