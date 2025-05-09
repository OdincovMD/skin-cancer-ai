import os
from typing import Tuple, Any

import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

#переименовано с several_reticular_lines_simm_assimm.pth -> several_lines_reticular
MODEL_PATH = os.path.join('weight', 'several_lines_reticular.pth')


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """
    Preprocesses an image for prediction by converting it from BGR to RGB,
    converting to tensor, and normalizing it.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.functional.to_tensor(img)
    img = transforms.functional.normalize(img, MEAN, STD)
    return img.unsqueeze(0)


def load_model(checkpoint_path: str = MODEL_PATH, num_classes: int = 2) -> Tuple[EfficientNet, torch.device]:
    """
    Loads a pretrained EfficientNet model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        num_classes (int): The number of output classes for the model.

    Returns:
        Tuple[EfficientNet, torch.device]: A tuple containing the loaded model and the device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, device


_model_several_lines_reticular = None
_device_several_lines_reticular = None


def get_model() -> Any:
    """
    Retrieve the model, loading it from the file if it has not been loaded yet.

    Returns:
        Any: The loaded model object.
    """
    global _model_several_lines_reticular
    global _device_several_lines_reticular
    if not _model_several_lines_reticular and not _device_several_lines_reticular:
        _model_several_lines_reticular, _device_several_lines_reticular = load_model()
    return _model_several_lines_reticular, _device_several_lines_reticular


def main(img: np.ndarray) -> str:
    """
    Predicts the class of an image using a trained model.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        str: "Ассиметричные" or "Симметричные".
    """
    model, device = get_model()
    img = preprocess_image(img)
    img = img.to(device)
    with torch.no_grad():
        prediction = model(img)
    return "Ассиметричные" if torch.argmax(prediction) == 0 else "Симметричные"
