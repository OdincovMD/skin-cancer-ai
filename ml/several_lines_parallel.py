import os
from typing import Tuple

import cv2
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DATA_TRANSFORMS = {
    'test': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}

#переименован several_parallel_lines_borozd_grebesh.pth -> several_lines_parallel.pth
MODEL_PATH = os.path.join('weight', 'several_lines_parallel.pth')


def load_model(model_path: str = MODEL_PATH) -> Tuple[EfficientNet, torch.device]:
    """
    Loads a model from a specified checkpoint path.

    Args:
        model_path (str): Path to the model checkpoint file.

    Returns:
        Tuple[EfficientNet, torch.device]: Loaded model and the device it will be executed on.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device


_model_several_lines_parallel = None
_device_several_lines_parallel = None


def get_model():
    """
    Retrieve the model, loading it from the file if it has not been loaded yet.

    Returns:
        Any: The loaded model object.
    """
    global _model_several_lines_parallel
    global _device_several_lines_parallel
    if not _model_several_lines_parallel and not _device_several_lines_parallel:
        _model_several_lines_parallel, _device_several_lines_parallel = load_model()
    return _model_several_lines_parallel, _device_several_lines_parallel


# до этого main была функцией для вызова двух функций
def main(img: np.ndarray) -> str:
    """
    The main function to conduct the entire process of loading, preprocessing, and predicting an image.

    Args:
        img (np.ndarray): image for preprocessing.

    Returns:
        str: "Борозды" or "Гребешки"
    """
    model, device = get_model()
    # До этого это была функция для обработки
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img, dtype=torch.float) / 255.0
    img = transforms.ToPILImage()(img.permute(1, 2, 0).to(torch.uint8).numpy())
    image_transform = DATA_TRANSFORMS['test'](img).unsqueeze(0)

    # раньше была функция для предсказания
    with torch.no_grad():
        prediction = model(image_transform.to(device))

    return "Борозды" if torch.argmax(prediction) == 0 else "Гребешки"
