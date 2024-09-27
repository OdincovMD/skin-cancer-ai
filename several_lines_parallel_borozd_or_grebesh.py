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


def load_model(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img, dtype=torch.float) / 255.0
    img = transforms.ToPILImage()(img.permute(1, 2, 0).to(torch.uint8).numpy())
    img = data_transforms['test'](img).unsqueeze(0)
    return img


def predict(model, device, image_transform):
    with torch.no_grad():
        prediction = model(image_transform.to(device))
    return "Borozd" if torch.argmax(prediction) == 0 else "Grebesh"


model, device = load_model('weight/several_parallel_lines_borozd_grebesh.pth')


def main(img):
    image_transform = preprocess_image(img)
    prediction = predict(model, device, image_transform)
    return prediction


if __name__ == "__main__":
    img_path = '26.jpg'
    img = cv2.imread(img_path)
    prediction = main(img)
    print(prediction)
