import torch
import cv2
import numpy as np
from torchvision import models, transforms

IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])  


def load_model(model_name: str, model_path: str) -> torch.nn.Module:
    """
    Function for loading model weights

    Parameters
    ____________
        image : np.ndarray
            Original image of neoplasm
        model_path : str
            Path to load model from

    Returns
    ____________
        model : torch.nn.Module
            Loaded classification model
    """
    model = getattr(models, model_name)(weights=None, num_classes=1)
    model.load_state_dict(torch.load(f=model_path, map_location=torch.device(DEVICE)))
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Function for preprocessing image

    Parameters
    ____________
        image : np.ndarray
            Original image of neoplasm

    Returns
    ____________
        torch.Tensor
            Transformed image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = TRANSFORM(image)
    return image_tensor.unsqueeze(0).to(DEVICE, non_blocking=True)


def predict(model: torch.nn.Module, image_tensor: torch.Tensor) -> str:
    """
    Function for performing classification of neoplasm

    Parameters
    ____________
        model : torch.nn.Module
            Model used for classification
        image_tensor : torch.Tensor
            Transformed original image of neoplasm

    Returns
    ____________
        label : str
            Color of reticular lines according to classificator
    """
    output = model(image_tensor)
    prediction = torch.sigmoid(output) >= 0.5
    label = "Черные" if prediction else "Коричневые"
    return label


model_path = "weight/one_lines_reticular_one_color.pth"
model_name = "resnet50"

model = load_model(model_name, model_path)


def main(image: np.ndarray) -> str:
    """
   Classification of reticular lines presented in one color by color:
   brown or black

   Parameters
   ____________
       image : np.ndarray
           Original image of neoplasm

   Returns
   ____________
       result : str
           Color of reticular lines according to classificator
           Черные, Коричневые
   """
    image_tensor = preprocess_image(image)
    result = predict(model, image_tensor)
    return result



