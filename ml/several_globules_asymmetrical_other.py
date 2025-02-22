import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import transforms
from resizeimage import resizeimage
from PIL import Image
from typing import Any

WEIGHT_PATH = r'weight/several_globules_asymmetrical_other.pth'

LABELS = {
    0: 'Жёлтый или белый',
    1: 'Оранжевый',
    2: 'Красный или пурпурный'
}

_model_several_globules_asymmetrical_orher = None

class FourChannelClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(FourChannelClassifier, self).__init__()
        
        # Первый свёрточный блок
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Сверточные блоки (ResNet-like)
        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Adaptive Pooling + Fully Connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(self._basic_block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(self._basic_block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _basic_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def load_model(model_path: str = WEIGHT_PATH) -> FourChannelClassifier:
    """
    Loads a model from a specified checkpoint path.

    Args:
        model_path (str): Path to the model checkpoint file.

    Returns:
        Loaded model and the device it will be executed on.
    """
    model = FourChannelClassifier(num_classes=3)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu')), 
        strict=False
        )
    model.eval()
    return model

def get_model() -> Any:
    """
    Retrieve the model, loading it from the file if it has not been loaded yet.

    Returns:
        Any: The loaded model object.
    """
    global _model_several_globules_asymmetrical_orher
    if not _model_several_globules_asymmetrical_orher:
        _model_several_globules_asymmetrical_orher = load_model()
    return _model_several_globules_asymmetrical_orher

def main(img: np.ndarray, mask: np.ndarray) -> str:
    """
    Main function for classifying an image as Жёлтый или белый, Оранжевый, Красный или пурпурный.

    Args:
        img (np.ndarray): Image array in NumPy format.
        mask (np.ndarray): Mask for the image.

    Returns:
        str: Classification result: "Жёлтый или белый" or "Оранжевый" or "Красный или пурпурный".
"""

    # Трансформации изображения
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    resize = transforms.Lambda(
        lambda img:  resizeimage.resize_cover(img, [256, 256], validate=False)
    )

    to_tensor =  transforms.ToTensor()

    transform_mask = transforms.Compose([
            resize,
            to_tensor,
        ])
    transform_image = transforms.Compose([
            resize,
            to_tensor,
            normalize
        ])
    
    image = Image.fromarray(img[:, :, ::-1]).convert("RGB")  # Загружаем изображение как RGB
    mask = Image.fromarray(mask).convert("L")    # Загружаем маску как 8-битное изображение
    image = transform_image(image)
    mask = transform_mask(mask)

    # Преобразуем маску в бинарный канал
    mask = (mask > 0).float()  # Маска становится бинарной (0 или 1)
    # Объединяем изображение и маску в один тензор
    combined = torch.cat([image, mask], dim=0)  # Добавляем маску как 4-й канал
    combined = combined.to(torch.device('cpu'))
    combined = combined.unsqueeze(0)  # Добавляем канал
    
    model = get_model()
    output = model(combined)
    probs = F.softmax(output, dim=1)  # Применяем softmax
    # Предсказанные классы
    preds = torch.argmax(probs, dim=1)
    return LABELS[preds.item()]
