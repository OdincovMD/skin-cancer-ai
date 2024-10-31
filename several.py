import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

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

def load_model(path_to_model):
    CNNmodel = ConvolutionalNetwork()
    state = torch.load(path_to_model, map_location=device)
    CNNmodel.load_state_dict(state)
    CNNmodel.to(device)
    CNNmodel.eval()
    return CNNmodel

path_to_model = 'weight/ManyPatterns_DominantPattern.pt'
CNNmodel = load_model(path_to_model)

def predict(image_np: np.ndarray) -> str:
    image = Image.fromarray(image_np)
    transform = transforms.Compose([
        transforms.Resize(990),
        transforms.CenterCrop(990),
        transforms.ToTensor()
    ])
    input_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = CNNmodel(input_image)
    sm = nn.Softmax(dim=1)
    prediction = sm(y_pred)
    predicted_index = torch.argmax(prediction, dim=1).item()
    classes = {0: 'Комки', 1: 'Круги', 2: 'Линии', 3: 'Точки'}
    predicted_class = classes[predicted_index]
    return predicted_class

def main(image_np: np.ndarray):
    return predict(image_np)

if __name__ == '__main__':
    path_to_image = '26.jpg'
    image_np = np.array(Image.open(path_to_image))
    print(main(image_np))