import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os

class CustomNeuralNetResNet(torch.nn.Module):
    def __init__(self, outputs_number):
        super(CustomNeuralNetResNet, self).__init__()
        self.net = models.resnet152(pretrained=True)

        # Выключаем переобучение весов каждого слоя модели, кроме последнего
        for param in self.net.parameters():
            param.requires_grad = False

        TransferModelOutputs = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(TransferModelOutputs, outputs_number)
        )

    def forward(self, x):
        return self.net(x)

def main(img: np.ndarray, mask: np.ndarray) -> str:
    """
    Main function for classifying an image as symmetrical or asymmetrical.

    Args:
        img (np.ndarray): Image array in NumPy format.
        mask (np.ndarray): Mask for the image.

    Returns:
        str: Classification result: "Симметрия" or "Асимметрия".
"""

    # Трансформации изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(200),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mask= np.stack([mask] * 3, axis=-1)
    masked_img = np.where(mask == 255, img, 0)
    img = Image.fromarray(masked_img)
    img_tensor = transform(img).unsqueeze(0)


    model = CustomNeuralNetResNet(2)
    model.load_state_dict(
        torch.load(r'weight/cl_several_line_parallel_furrow_ridges_sym.pth', map_location=torch.device('cpu'))
        )
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).cpu().numpy()[0]

    return ['Асимметрия', 'Симметрия'][pred_class]