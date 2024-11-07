import torch
from torchvision.transforms import transforms
import numpy as np
from resizeimage import resizeimage
from PIL import Image


class CustomNeuralNetResNet(torch.nn.Module):
    """
    Custom neural network based on the ResNet50 wide architecture.

    Args:
        outputs_number (int): The number of output neurons.
    """

    def __init__(self, outputs_number):
        super(CustomNeuralNetResNet, self).__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        
        for param in self.net.parameters():
            param.requires_grad = False
        
        #Размораживаем 4-ый блок
        for param in self.net.layer4.parameters():
            param.requires_grad = True

        
        #Полносвязный слой (fc)
        TransferModelOutputs = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(TransferModelOutputs, 512),  #Промежуточный слой
            torch.nn.ReLU(),  #Нелинейность
            torch.nn.BatchNorm1d(512),  #Нормализация
            torch.nn.Dropout(0.6),  #Дроп-аут для регуляризации
            torch.nn.Linear(512, outputs_number)  #Выходной слой
        )

    def forward(self, x):
        return self.net(x)
    
def main(img: np.ndarray) -> str:
    """
    Main function for classifying an image by one or multiple features.

    Args:
        img (np.ndarray): Image array in numpy format.

    Returns:
        str: Classification result: "Несколько" or "Один".
    """

    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: resizeimage.resize_cover(img, [288, 288], validate=False)
            ),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda img: torch.randn_like(img)*0.02 + img
                ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ]
    )
    
    img = Image.fromarray(img)
    img_tensor = transform(img).unsqueeze(0)

    models = []
    for fold in range(4):
        model = CustomNeuralNetResNet(2)
        model.load_state_dict(torch.load(f'weight/best_model_fold_{fold+1}.pth', 
                                         map_location=torch.device('cpu'))) 
        models.append(model)

    voting = 'soft'

    fold_preds = []
    fold_probs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            fold_probs.append(probs.cpu().numpy())  # Для soft voting
            pred_class = probs.argmax(dim=1).cpu().numpy()[0]
            fold_preds.append(pred_class)

    # Majority voting: выбор класса с наибольшим количеством голосов
    if voting == 'majority':
        final_pred_class = np.bincount(fold_preds).argmax()

    # Soft voting: усреднение вероятностей классов
    elif voting == 'soft':
        avg_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
        final_pred_class = avg_probs.argmax(axis=1)[0]

    return ['Несколько', 'Один'][final_pred_class]