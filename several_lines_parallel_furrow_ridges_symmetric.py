import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
import numpy as np
from torchvision import transforms, models
from tqdm import tqdm
from typing import Optional, Callable, Any, List
import os

os.chdir(os.path.dirname(__file__))


class CustomNeuralNetResNet(torch.nn.Module):
    """
    Кастомная нейронная сеть, основанная на архитектуре ResNet50.

    Аргументы:
        outputs_number (int): Количество выходных нейронов.

    Атрибуты:
        net (torch.nn.Module): Загруженная модель ResNet50 с замененным последним слоем.
    """

    def __init__(self, outputs_number: int) -> None:
        super(CustomNeuralNetResNet, self).__init__()
        self.net = models.resnet50(pretrained=True)

        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.net.layer4.parameters():
            param.requires_grad = True

        TransferModelOutputs = self.net.fc.in_features

        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(TransferModelOutputs, outputs_number)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


some_line_model = CustomNeuralNetResNet(2)
some_line_model.load_state_dict(torch.load(
    r'several_line_parallel_furrow_ridges_sym.pth', map_location=torch.device('cpu')))
some_line_model.eval()


class NumpyImageDataset(Dataset):
    """
    Датасет для работы с изображениями в формате массива NumPy.

    Атрибуты:
        image_array (np.ndarray): Массив изображений.
        transform (callable): Преобразование данных, которое применяется к изображению.

    Методы:
        __init__(self, image_array, transform=None): Конструктор класса.
        __len__(self): Возвращает количество элементов в датасете.
        __getitem__(self, idx): Возвращает элемент датасета по индексу.
    """

    def __init__(self, image_array: np.ndarray, transform: Optional[Callable[[Any], Any]] = None) -> None:
        self.image_array = image_array
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_array)

    def __getitem__(self, idx: int) -> Any:
        image = self.image_array[idx]
        if self.transform:
            image = self.transform(image)
        return image


def main(img: np.ndarray) -> str:
    """
    Основная функция для классификации изображения на симметрию и асимметрию.

    Args:
        img (np.ndarray): Массив изображения в формате numpy.

    Returns:
        str: Результат классификации: "Симметрия" или "Асимметрия".
    """

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(200),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_array_list = [img]
    numpy_image_dataset = NumpyImageDataset(
        image_array_list, transform=transform)

    dataloader = DataLoader(numpy_image_dataset, batch_size=1, shuffle=False)

    info = ["Симметрия", "Асимметрия"]

    test_predictions = []
    for inputs in tqdm(dataloader):
        with torch.set_grad_enabled(False):
            preds = some_line_model(inputs)
        test_predictions.append(
            torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())
    pred = 0
    if test_predictions[0] < 0.90:
        pred = 1

    return info[pred]


if __name__ == "__main__":
    img = mpimg.imread("26.jpg")  # type(img) is numpy.ndarray
    result = main(img)
    print(result)
