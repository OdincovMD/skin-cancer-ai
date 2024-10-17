import cv2
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from typing import Tuple
import numpy as np

INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# глобальная по сути константа, должна быть с большой буквы
DATA_TRANSFORMS = {
    'test': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}


def load_model(model_path: str) -> Tuple[EfficientNet, torch.device]:
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


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """
    Processes an image for prediction with the model.

    Args:
        img (np.ndarray): Original image in BGR.

    Returns:
        torch.Tensor: Image tensor prepared for input to the model.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img, dtype=torch.float) / 255.0
    img = transforms.ToPILImage()(img.permute(1, 2, 0).to(torch.uint8).numpy())
    img = DATA_TRANSFORMS['test'](img).unsqueeze(0)
    return img


def predict(model: EfficientNet, device: torch.device, image_transform: torch.Tensor) -> str:
    """
    Makes a prediction on the class of the image using the model.

    Args:
        model (EfficientNet): The loaded model.
        device (torch.device): Device for computation.
        image_transform (torch.Tensor): Processed image tensor.

    Returns:
        str: Predicted class name.
    """
    with torch.no_grad():
        prediction = model(image_transform.to(device))
    return "Borozd" if torch.argmax(prediction) == 0 else "Grebesh"


model, device = load_model('weight/several_parallel_lines_borozd_grebesh.pth')
# если это опять глобальное, то почему мы с lower case обозвали переменную 
# и еще и не передаем ее в функцию. BAd
# TODO: сделать из этого global и передавать в функцию или сделать функцию на загрузку модели (как будто муторно)
# TODO: переименовать весовой файл в соответствии с названием модуля -> several_lines_parallel_borozd_or_grebesh.pth

def main(img: np.ndarray) -> str:
    """
    The main function to conduct the entire process of loading, preprocessing, and predicting an image.

    Args:
        img (np.ndarray): image for preprocessing.

    Returns:
        str: Result of the model prediction.
    """
    image_transform = preprocess_image(img)
    prediction = predict(model, device, image_transform)
    return prediction


if __name__ == "__main__":
    img_path = '26.jpg'
    img = cv2.imread(img_path)
    prediction = main(img)
    print(prediction)
