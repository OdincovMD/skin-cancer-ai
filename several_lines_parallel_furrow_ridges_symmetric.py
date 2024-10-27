import os
from typing import Optional, Callable, Any

import matplotlib.image as mpimg # TODO: убрать после удаления загрузки изображения для тестрирования
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

os.chdir(os.path.dirname(__file__))
MODEL_PATH = os.path.join('weight','several_line_parallel_furrow_ridges_sym.pth')

#TODO Вопрос: img = mpimg.imread("26.jpg") - RGB order. Соответственно влияет на каналы, которые подаются в нейронку на вход. при чтении cv2 - там bgr каналы

class CustomNeuralNetResNet(torch.nn.Module):
    """
    A custom neural network based on the ResNet50 architecture.

    Args:
        outputs_number (int): Number of output neurons.

    Returns:
        net (torch.nn.Module): Loaded ResNet50 model with modified final layer.
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


def load_model() -> CustomNeuralNetResNet:
    model = CustomNeuralNetResNet(2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model


_model_several_lines_parallel_furrow_edges_symmetric = None

def get_model():
    global _model_several_lines_parallel_furrow_edges_symmetric
    if not _model_several_lines_parallel_furrow_edges_symmetric:
        _model_several_lines_parallel_furrow_edges_symmetric = load_model()
    return _model_several_lines_parallel_furrow_edges_symmetric


class NumpyImageDataset(Dataset):
    """
    A dataset class for handling images in the form of NumPy arrays.

    Args:
        image_array (np.ndarray): Array of images.
        transform (callable): A callable that applies data transformations to the image.

    Methods:
        init(self, image_array, transform=None): Class constructor.
        len(self): Returns the number of elements in the dataset.
        getitem(self, idx): Returns a dataset item by index.
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
    The main function for classifying an image as symmetric or asymmetric.

    Args:
        img (np.ndarray): Image array in numpy format.

    Returns:
        str: Classification result: "Симметрия" or "Асимметрия".
    """
    model = get_model()
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
            preds = model(inputs)
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
