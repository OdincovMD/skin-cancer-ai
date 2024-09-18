import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

PATH_TO_MODEL = "weight/several_clumps_asymmetrical.pt"
SIZE = 400
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def segment_image(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    if h == SIZE and w == SIZE:
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

    nevus = [cnt for cnt in nevus if cv2.boundingRect(cnt) not in [(1, 1), (2559, 1), (1, 1919), (2559, 1919)]]

    if len(nevus) > 1:
        areas = [cv2.contourArea(cnt) for cnt in nevus]
        max_area = max(areas)
        nevus = [cnt for cnt, area in zip(nevus, areas) if area == max_area]

    if not nevus:
        return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    cnt = nevus[0]
    x, y, w, h = cv2.boundingRect(cnt)
    xdiff = (h - w) // 2 if h > w else 0
    ydiff = (w - h) // 2 if w > h else 0

    ram = 100
    x1, y1 = max(1, x - ram - xdiff), max(1, y - ram - ydiff)
    x2, y2 = min(2559, x + w + ram + xdiff), min(1919, y + h + ram + ydiff)

    xsize, ysize = y2 - y1, x2 - x1

    if xsize < SIZE or ysize < SIZE:
        xcorr = (SIZE + 2 - xsize) // 2 if xsize < SIZE else 0
        ycorr = (SIZE + 2 - ysize) // 2 if ysize < SIZE else 0
        x1, x2 = max(1, x1 - xcorr), min(2559, x2 + xcorr)
        y1, y2 = max(1, y1 - ycorr), min(1919, y2 + ycorr)

    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def create_model() -> nn.Sequential:
    model = models.vgg16(pretrained=True)
    model = nn.Sequential(*(list(model.children())[:-2]))

    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(12 * 12 * 512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
        nn.Softmax(dim=1)
    )

    model.classifier = classifier
    model.to(DEVICE)

    return model


model = create_model()
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=DEVICE))
model.to(DEVICE)


def main(image: np.ndarray) -> str:
    image = segment_image(image)

    transform = transforms.Compose([transforms.ToTensor()])
    input_image = transform(Image.fromarray(image))
    input_image = input_image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_image)
        probabilities = nn.Softmax(dim=1)(output)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    classes = {0: 'Другой', 1: 'Меланин'}
    return classes[predicted_class]


if __name__ == "__main__":
    path_to_image = "26.jpg"
    image_np = np.array(Image.open(path_to_image))
    print(main(image_np))
