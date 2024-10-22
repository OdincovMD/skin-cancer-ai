import os

import numpy as np
import torch
import cv2
from torchvision import models, transforms
import requests
import base64

MODEL_PATH = "weight/several_globules_symmetricOrAsymmetric.pth"
MODEL_NAME = "resnet50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = getattr(models, MODEL_NAME)(weights=None, num_classes=1)
MODEL.load_state_dict(torch.load(f=MODEL_PATH, map_location=torch.device(DEVICE)))
MODEL = MODEL.to(DEVICE)
MODEL.eval()


def detect_globs(image: np.ndarray) -> np.ndarray:
    """
    Detects globules in the input image using the SimpleBlobDetector from OpenCV and
    returns globules mask.
    :param image: Input image.
    :return: Binary mask indicating the detected globules.
    """
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.minDistBetweenBlobs = 0.001
    params.minThreshold = 0
    params.filterByArea = True
    params.minArea = 50
    params.filterByCircularity = True
    params.minCircularity = 0.001
    params.filterByConvexity = True
    params.minConvexity = 0.001
    params.filterByInertia = True
    params.minInertiaRatio = 0.001

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)

    mask = np.zeros_like(image, dtype=np.uint8)
    for kp in keypoints:
        cv2.circle(mask, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), (255, 255, 255), -1)

    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    globs_mask = detect_globs(image)
    globs_mask = np.expand_dims(globs_mask, axis=2).repeat(3, axis=2)
    segmented_image = cv2.bitwise_or(image, globs_mask)

    segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=mask)

    return segmented_image


def predict_one_image(image: np.ndarray) -> str:
    image_size = 256  # model was learned on this image size
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.to(DEVICE, non_blocking=True)
    output = MODEL(image_tensor.unsqueeze(0))
    prediction = torch.sigmoid(output) >= 0.5
    label = "АСИММЕТРИЧНЫЕ" if prediction else "СИММЕТРИЧНЫЕ"

    return label


def main(image: np.ndarray, mask: np.ndarray) -> str:
    """
    You need to use only NEO without background.
    :param image: Input image as a NumPy array.
    :return: Predicted label ("АСИММЕТРИЧНЫЕ" or "СИММЕТРИЧНЫЕ").
    """
    image = apply_mask(image, mask)
    return predict_one_image(image)


# if __name__ == "__main__":
#     image_path = "26.jpg"
#     image = cv2.imread(image_path)

#     rf = Roboflow(api_key="GmJT3lC4NInRGZJ2iEit")
#     project = rf.workspace("neo-dmsux").project("neo-v6wzn")
#     model = project.version(2).model

#     data = model.predict("26.jpg").json()
#     width = data['predictions'][0]['image']['width']
#     height = data['predictions'][0]['image']['height']

#     encoded_mask = data['predictions'][0]['segmentation_mask']
#     mask_bytes = base64.b64decode(encoded_mask)
#     mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
#     mask_image = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
#     mask = np.where(mask_image == 1, 255, mask_image)
#     mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

#     result = main(image, mask)
#     print(result)
