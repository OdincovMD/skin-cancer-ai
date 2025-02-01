import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image

WEIGHT_PATH = r'weight/several_lines_parallel_furrow.pth'

LABELS = {0: 'Асимметрия', 
         1: 'Симметрия'}

_model_several_lines_parallel_furrow = None


class CustomNeuralNetResNet(torch.nn.Module):
    """
    A custom neural network based on the ResNet50 architecture.

    Args:
        outputs_number (int): Number of output neurons.

    Returns:
        net (torch.nn.Module): Loaded ResNet50 model with modified final layer.
    """
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
    

def load_model(model_path: str = WEIGHT_PATH) -> CustomNeuralNetResNet:
    """
    Loads a model from a specified checkpoint path.

    Args:
        model_path (str): Path to the model checkpoint file.

    Returns:
        Tuple[EfficientNet, torch.device]: Loaded model and the device it will be executed on.
    """
    model = CustomNeuralNetResNet(2)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))
        )
    model.eval()
    return model


def get_model():
    """
    Retrieve the model, loading it from the file if it has not been loaded yet.

    Returns:
        Any: The loaded model object.
    """
    global _model_several_lines_parallel_furrow
    if not _model_several_lines_parallel_furrow:
        _model_several_lines_parallel_furrow = load_model()
    return _model_several_lines_parallel_furrow



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
    masked_img = np.where(mask == 255, img[:, :, ::-1], 0)
    img = Image.fromarray(masked_img)
    img_tensor = transform(img).unsqueeze(0)

    model = get_model()

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).cpu().numpy()[0]

    return LABELS[pred_class]
