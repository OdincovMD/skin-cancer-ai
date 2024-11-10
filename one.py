# код @kiwifm

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

PATH_TO_MODEL = "weight/one.pt"
IMAGE_SIZE = 400
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_model() -> nn.Sequential:

    """
    Creates a model of the required architecture.
    
    Returns:
        model (torch.nn.Sequential): The model of the required architecture for class prediction.
    """

    model = models.vgg16(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-2]))

    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(12 * 12 * 512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 5),
        nn.Softmax(dim=1)
    )

    model.classifier = classifier
    model.to(DEVICE)
    return model


MODEL = create_model()
STATE = torch.load(PATH_TO_MODEL, map_location=torch.device(DEVICE))
MODEL.load_state_dict(STATE)
MODEL.to(DEVICE)


# В дальнейшем при разработке можно переобучить модель на изображениях с маской, тогда можно оставить только обрезку до размера 400*400
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Finding a mole in the image and reducing it to a size of 400*400.
    
    Paraneters:
        img (np.ndarray): The input image.
    Returns:
        np.ndarray: The image after preprocessing.
    """
    h, w, _ = img.shape
    if h == IMAGE_SIZE and w == IMAGE_SIZE:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    labels = np.unique(markers)

    nevus = []
    for label in labels[2:]:
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nevus.append(contours[0])

    nevus = [cnt for cnt in nevus if not (cv2.boundingRect(cnt)[:2] == (1, 1) or
                                          cv2.boundingRect(cnt)[2:] == (2559, 1919))]

    if len(nevus) > 1:
        areas = [cv2.contourArea(cnt) for cnt in nevus]
        nevus = [nevus[areas.index(max(areas))]]

    if not nevus:
        return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    x, y, w, h = cv2.boundingRect(nevus[0])
    xdiff = (h - w) // 2 if h > w else 0
    ydiff = (w - h) // 2 if w > h else 0

    margin = 100
    x1, y1 = max(1, x - margin - xdiff), max(1, y - margin - ydiff)
    x2, y2 = min(2559, x + w + margin + xdiff), min(1919, y + h + margin + ydiff)

    xsize, ysize = y2 - y1, x2 - x1
    if xsize < IMAGE_SIZE or ysize < IMAGE_SIZE:
        xcorr = (IMAGE_SIZE + 2 - xsize) // 2 if xsize < IMAGE_SIZE else 0
        ycorr = (IMAGE_SIZE + 2 - ysize) // 2 if ysize < IMAGE_SIZE else 0
        x1, x2 = max(1, x1 - xcorr), min(2559, x2 + xcorr)
        y1, y2 = max(1, y1 - ycorr), min(1919, y2 + ycorr)

    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def main(image: np.ndarray) -> str:
    """
    Calls the image preprocessing and model creation functions, and then passes the image to the model for prediction.

    Paraneters:
        img (np.ndarray): The input image.    
    Returns:
        classes[predicted_index] (str): The label of the predicted class: "Бесструктурная область", "Комки", "Круги", "Линии" or "Точки".
    """
    image = preprocess_image(image)
    transform = transforms.Compose([transforms.ToTensor()])
    input_image = transform(Image.fromarray(image))

    input_image = torch.FloatTensor(input_image).to(DEVICE).unsqueeze(0)
    prediction = MODEL(input_image)
    predicted_index = torch.argmax(prediction, dim=1).item()

    classes = {0: 'Бесструктурная область', 1: 'Комки', 2: 'Круги', 3: 'Линии', 4: 'Точки'}
    return classes[predicted_index]
