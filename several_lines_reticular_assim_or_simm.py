import numpy as np
import cv2
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from typing import Tuple

INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# зачем вообще этот словарь, который нигде не используется??? я бы удалила
# TODO: согласовать удаление
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}


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


def load_model(checkpoint_path: str, num_classes: int = 2) -> Tuple[EfficientNet, torch.device]:
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


def predict(model: EfficientNet, device: torch.device, img: np.ndarray) -> str:
    """
    Predicts the class of an image using a trained model.

    Args:
        model (EfficientNet): Loaded model.
        device (torch.device): Computation device.
        img (np.ndarray): Input image in BGR format.

    Returns:
        str: Prediction result, either "Asymmetric" or "Symmetric".
    """
    img = preprocess_image(img)
    img = img.to(device)
    with torch.no_grad():
        prediction = model(img)
    return "Asymmetric" if torch.argmax(prediction) == 0 else "Symmetric"


model, device = load_model('weight/several_reticular_lines_simm_assimm.pth')
# Ура опять глобальные переменные в lowercase. Почему загрузка вне функций? засунуть в main наверное
# TODO: согласовать переименование весов в соответствии с названием модуля -> several_lines_reticular_assim_or_simm.pth

# эта функция не имеет смысла. Либо изменить предикт на main или наоборот. Только жрет ресурсы
def main(img: np.ndarray) -> str:
    return predict(model, device, img)


if __name__ == "__main__":
    img = cv2.imread('26.jpg')
    print(main(img))
