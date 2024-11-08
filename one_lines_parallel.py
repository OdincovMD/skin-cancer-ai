import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvolutionalNetwork(nn.Module):
    """
   This class builds a convolutional neural network consisting of
   2 convolutional layers and 3 fully-connected.
   """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(246 * 246 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 246 * 246 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

model_path = 'weight/OnePattern_Lines_ParallelLines_TypeOfLines.pt'
CNNmodel = ConvolutionalNetwork()

state = torch.load(model_path, map_location=device)
CNNmodel.load_state_dict(state)
CNNmodel.to(device)


class AddGaussianNoise:
    """
    Class of Gaussian noise for transformation of image to classify
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean


def classify_image(image_np):
    """
    Function for performing classification of neoplasm

    Parameters
    ____________
        image_np : np.ndarray
            Original image of neoplasm

    Returns
    ____________
        classes[predicted_class] : str
            Pattern of parallel lines according to classificator
    """
    transform = transforms.Compose([
        transforms.Resize(990),
        transforms.CenterCrop(990),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.1),
        transforms.RandomPerspective(distortion_scale=0.1)
    ])

    image = Image.fromarray(image_np)
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = CNNmodel(input_image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    classes = {0: 'Борозды', 1: 'Гребешки', 2: 'Пересекающиеся гребешки и борозды'}
    return classes[predicted_class]

def main(image_np):
    """
    Classification of parallel lines by the type:
    Гребешки, Борозды, Пересекающиеся гребешки и борозды

    Parameters
    ____________
        image_np : np.ndarray
            Original image of neoplasm

    Returns
    ____________
        predicted_class : str
            Pattern of parallel lines according to classificator
            Гребешки, Борозды, Пересекающиеся гребешки и борозды
    """
    predicted_class = classify_image(image_np)
    return predicted_class

