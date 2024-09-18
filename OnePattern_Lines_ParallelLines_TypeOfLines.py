import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvolutionalNetwork(nn.Module):
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
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

def augmentation(train_data):
    transform_noise = transforms.Compose([
        transforms.Resize(990),
        transforms.CenterCrop(990),
        AddGaussianNoise(0., 0.1),
        transforms.RandomPerspective(distortion_scale=0.1)
    ])

    augmented_data = []
    for image, label in train_data:
        num_augmentations = 4 if label == 0 else 10
        for _ in range(num_augmentations):
            augmented_image = transform_noise(image)
            augmented_data.append((augmented_image, label))
    return augmented_data

def classify_image(image_np):
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
    predicted_class = classify_image(image_np)
    return predicted_class

if __name__ == '__main__':
    path_to_image = '26.jpg'
    image_np = np.array(Image.open(path_to_image))
    print(main(image_np))