import copy
import torch
from torchvision import models
from torchvision.transforms import transforms
import numpy as np
from resizeimage import resizeimage
from PIL import Image
from typing import List, Optional

# Shared ImageNet backbone — torch.hub.load is expensive; must not run per-request or per-fold.
_HUB_WIDE_RESNET_TEMPLATE: Optional[torch.nn.Module] = None


def _get_hub_wide_resnet_template() -> torch.nn.Module:
    global _HUB_WIDE_RESNET_TEMPLATE
    if _HUB_WIDE_RESNET_TEMPLATE is None:
        _HUB_WIDE_RESNET_TEMPLATE = models.wide_resnet50_2(weights=None)
    return _HUB_WIDE_RESNET_TEMPLATE


def _build_net_from_template(outputs_number: int) -> torch.nn.Module:
    net = copy.deepcopy(_get_hub_wide_resnet_template())
    for param in net.parameters():
        param.requires_grad = False
    for param in net.layer4.parameters():
        param.requires_grad = True
    tin = net.fc.in_features
    net.fc = torch.nn.Sequential(
        torch.nn.Linear(tin, 512),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(512),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(512, outputs_number),
    )
    return net


class CustomNeuralNetResNet(torch.nn.Module):
    """
    Custom neural network based on the ResNet50 wide architecture.

    Args:
        outputs_number (int): The number of output neurons.
    """

    def __init__(self, outputs_number):
        super(CustomNeuralNetResNet, self).__init__()
        self.net = _build_net_from_template(outputs_number)

    def forward(self, x):
        return self.net(x)


_ENSEMBLE_MODELS: Optional[List[CustomNeuralNetResNet]] = None


def _get_ensemble() -> List[CustomNeuralNetResNet]:
    global _ENSEMBLE_MODELS, _HUB_WIDE_RESNET_TEMPLATE
    if _ENSEMBLE_MODELS is None:
        built: List[CustomNeuralNetResNet] = []
        for fold in range(5):
            m = CustomNeuralNetResNet(2)
            m.load_state_dict(
                torch.load(
                    f"weight/one_several_{fold + 1}.pth",
                    map_location=torch.device("cpu"),
                )
            )
            m.eval()
            built.append(m)
        _ENSEMBLE_MODELS = built
        _HUB_WIDE_RESNET_TEMPLATE = None
    return _ENSEMBLE_MODELS


def main(img: np.ndarray, mask: np.ndarray) -> str:
    """
    Main function for classifying an image by one or multiple features.

    Args:
        img (np.ndarray): Image array in numpy format.
        mask (np.ndarray): Mask for the image.

    Returns:
        str: Classification result: "Несколько признаков" or "Один признак".
    """

    mask = np.stack([mask] * 3, axis=-1)
    masked_img = np.where(mask == 255, img[:, :, ::-1], 0)
    img = Image.fromarray(masked_img)

    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: resizeimage.resize_cover(img, [288, 288], validate=False)
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda img: torch.randn_like(img) * 0.02 + img
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    img_tensor = transform(img).unsqueeze(0)

    voting = "soft"

    fold_preds = []
    fold_probs = []

    for model in _get_ensemble():
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            fold_probs.append(probs.cpu().numpy())  # Для soft voting
            pred_class = probs.argmax(dim=1).cpu().numpy()[0]
            fold_preds.append(pred_class)

    # Majority voting: выбор класса с наибольшим количеством голосов
    if voting == "majority":
        final_pred_class = np.bincount(fold_preds).argmax()

    # Soft voting: усреднение вероятностей классов
    elif voting == "soft":
        avg_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
        final_pred_class = avg_probs.argmax(axis=1)[0]

    return ["Несколько признаков", "Один признак"][final_pred_class]
