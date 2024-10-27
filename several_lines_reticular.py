import numpy as np
import cv2
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.functional.to_tensor(img)
    img = transforms.functional.normalize(img, MEAN, STD)
    return img.unsqueeze(0)


def load_model(checkpoint_path, num_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, device


def predict(model, device, img):
    img = preprocess_image(img)
    img = img.to(device)
    with torch.no_grad():
        prediction = model(img)
    return "Asymmetric" if torch.argmax(prediction) == 0 else "Symmetric"


model, device = load_model('weight/several_reticular_lines_simm_assimm.pth')


def main(img):
    return predict(model, device, img)


if __name__ == "__main__":
    img = cv2.imread('26.jpg')
    print(main(img))
