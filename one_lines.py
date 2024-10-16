import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 299
    return model_ft, input_size

# Загрузка модели в глобальную зону видимости
MODEL_PATH = 'weight/one_lines.pth'
NUM_CLASSES = 4
FEATURE_EXTRACT = True
model_ft, input_size = None, None

if os.path.exists(MODEL_PATH):
    model_ft, input_size = initialize_model(NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
    model_ft.load_state_dict(torch.load(MODEL_PATH))

def get_transformations():
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def evaluate(model, input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input)
    probs = nn.functional.softmax(output[0], dim=0)
    res_class = probs.argmax().item()
    return res_class

def main(img_to_classify):
    global model_ft, input_size

    if model_ft is None:
        model_ft, input_size = initialize_model(NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
        model_ft.load_state_dict(torch.load(MODEL_PATH))

    img_to_classify = cv2.cvtColor(img_to_classify, cv2.COLOR_BGR2RGB)
    img_transforms = get_transformations()
    clf_input = img_transforms(img_to_classify)
    input_batch = clf_input.unsqueeze(0)
    eval_res = evaluate(model_ft, input_batch)

    class_map = {0: 'Reticular', 1: 'Spread', 2: 'Parallel', 3: 'Curved'}
    result = class_map[eval_res]
    return result

if __name__ == '__main__':
    img_to_classify = cv2.imread('26.jpg')
    res = main(img_to_classify)
    print(res)