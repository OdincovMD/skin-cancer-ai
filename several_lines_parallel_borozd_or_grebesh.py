import cv2
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from typing import Tuple
import numpy as np

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

# TODO: переименовать весовой файл several_parallel_lines_borozd_grebesh.pth') в соответствии с названием модуля -> several_lines_parallel_borozd_or_grebesh.pth
MODEL, DEVICE = load_model('weight/several_parallel_lines_borozd_grebesh.pth')


# до этого main была функцией для вызова двух функций
def main(img: np.ndarray) -> str:
    """
    The main function to conduct the entire process of loading, preprocessing, and predicting an image.

    Args:
        img (np.ndarray): image for preprocessing.

    Returns:
        str: "Борозды" or "Гребешки"
    """
    # До этого это была функция для обработки
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img, dtype=torch.float) / 255.0
    img = transforms.ToPILImage()(img.permute(1, 2, 0).to(torch.uint8).numpy())
    image_transform = DATA_TRANSFORMS['test'](img).unsqueeze(0)

    # раньше была функция для предсказания
    with torch.no_grad():
        prediction = MODEL(image_transform.to(DEVICE))

    return "Борозды" if torch.argmax(prediction) == 0 else "Гребешки"


#выпилится после тестирования
if __name__ == "__main__":
    img_path = '26.jpg'
    img = cv2.imread(img_path)
    prediction = main(img)
    print(prediction)
