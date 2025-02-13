import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

path_to_model = 'weight/several.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(246 * 246 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 246 * 246 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


def main(image_np: np.ndarray) -> str:
    """
        Main function to predict dominant pattern in an image.
        Args:
            image_np (np.ndarray): Input image as a NumPy array.
        Returns:
            str: Predicted class label as a string - one out of: ['Комки', 'Круги', 'Линии', 'Точки'].
    """
    image = Image.fromarray(image_np)
    transform = transforms.Compose([
        transforms.Resize(990),
        transforms.CenterCrop(990),
        transforms.ToTensor()
    ])
    input_image = transform(image).unsqueeze(0).to(device)
    CNNmodel = ConvolutionalNetwork()
    state = torch.load(path_to_model, map_location=device)
    CNNmodel.load_state_dict(state)
    CNNmodel.to(device)
    CNNmodel.eval()
    with torch.no_grad():
        y_pred = CNNmodel(input_image)
    predicted_index = torch.argmax(y_pred, dim=1).item()
    classes = {0: 'Комки', 1: 'Круги', 2: 'Линии', 3: 'Точки'}
    predicted_class = classes[predicted_index]
    return predicted_class