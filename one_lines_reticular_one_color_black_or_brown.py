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
    model = getattr(models, model_name)(weights=None, num_classes=1)
    model.load_state_dict(torch.load(f=model_path, map_location=torch.device(DEVICE)))
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = TRANSFORM(image)
    return image_tensor.unsqueeze(0).to(DEVICE, non_blocking=True)


def predict(model: torch.nn.Module, image_tensor: torch.Tensor) -> str:
    output = model(image_tensor)
    prediction = torch.sigmoid(output) >= 0.5
    label = "ЧЕРНЫЕ" if prediction else "КОРИЧНЕВЫЕ"
    return label


model_path = "weight/single_lines_retic_oneColor_blackOrBrown.pth"
model_name = "resnet50"

model = load_model(model_name, model_path)


def main(image: np.ndarray) -> str:
    image_tensor = preprocess_image(image)
    result = predict(model, image_tensor)
    return result


# if __name__ == "__main__":
#     image_path = "26.jpg"
#     img = cv2.imread(image_path)
#     result = main(img)
#     print(result)
