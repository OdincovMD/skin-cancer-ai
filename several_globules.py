import os
import base64
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
from roboflow import Roboflow

# Constants
MODEL_PATH = os.path.join("weight", "several_globules_symmetricOrAsymmetric.pth")
MODEL_NAME = "resnet50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

def load_model():
    """
    Loads a pretrained ResNet model with specified weights and prepares it for use on the available device.

    Parameters:
        None

    Returns:
        torch.nn.Module: The loaded model, set to evaluation mode and ready for making predictions.
    """
    model = getattr(models, MODEL_NAME)(weights=None, num_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE),  weights_only=True))
    return model.to(DEVICE).eval()

MODEL = load_model()


def detect_globs(image: np.ndarray) -> np.ndarray:
    """
    Detects globules in the input image using OpenCV's SimpleBlobDetector and returns a binary mask.

    Parameters:
        image (np.ndarray): Input image as a numpy array.

    Returns:
        np.ndarray: Binary mask indicating the locations of detected globules.
    """
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = params.filterByArea = params.filterByCircularity = True
    params.filterByConvexity = params.filterByInertia = True
    params.minDistBetweenBlobs = 0.001
    params.minThreshold, params.minArea = 0, 50
    params.minCircularity = params.minConvexity = params.minInertiaRatio = 0.001

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)

    mask = np.zeros_like(image, dtype=np.uint8)
    for kp in keypoints:
        cv2.circle(mask, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), (255, 255, 255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies the globules mask and additional segmentation mask to the input image.

    Parameters:
        image (np.ndarray): Input image in numpy array format.
        mask (np.ndarray): Segmentation mask to apply to the image.

    Returns:
        np.ndarray: Image with applied masks, showing segmented regions.
    """
    globs_mask = detect_globs(image)
    globs_mask = np.expand_dims(globs_mask, axis=2).repeat(3, axis=2) 
    masked_image = cv2.bitwise_or(image, globs_mask)
    return cv2.bitwise_and(masked_image, masked_image, mask=mask)


def predict_symmetry(image: np.ndarray) -> str:
    """
    Predicts the symmetry label for a single image after processing with a pretrained model.

    Parameters:
        image (np.ndarray): Input image as a numpy array.

    Returns:
        str: Predicted label for the image, either "ASYMMETRIC" or "SYMMETRIC".
    """
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    image_tensor = transform(image=image)["image"].to(DEVICE)
    output = MODEL(image_tensor.unsqueeze(0))
    prediction = torch.sigmoid(output) >= 0.5
    return "АСИММЕТРИЧНЫЕ" if prediction else "СИММЕТРИЧНЫЕ"


def decode_segmentation_mask(data):
    """
    Decodes a base64-encoded segmentation mask from prediction data and converts it to a binary mask.

    Parameters:
        data (dict): Prediction data containing base64-encoded segmentation mask and image dimensions.

    Returns:
        np.ndarray: Decoded binary mask as a numpy array with values 0 and 255, matching image dimensions.
    """
    encoded_mask = data['predictions'][0]['segmentation_mask']
    mask_bytes = base64.b64decode(encoded_mask)
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask_image = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    return np.where(mask_image == 1, 255, mask_image)

def main(image_path: str) -> str:
    """
    Applies masking and performs symmetry classification on the input image.

    Parameters:
        image (np.ndarray): Input image as a numpy array.
        mask (np.ndarray): Segmentation mask for isolating relevant image areas.

    Returns:
        str: Predicted label indicating symmetry ("ASYMMETRIC" or "SYMMETRIC").
    """
    image = cv2.imread(image_path)

    rf = Roboflow(api_key="GmJT3lC4NInRGZJ2iEit")
    project = rf.workspace("neo-dmsux").project("neo-v6wzn")
    model = project.version(2).model
    data = model.predict(image_path).json()

    width, height = data['predictions'][0]['image']['width'], data['predictions'][0]['image']['height']
    mask = decode_segmentation_mask(data)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

    processed_image = apply_mask(image, mask)
    result = predict_symmetry(processed_image)
    return result


if __name__ == "__main__":
    image_path = "26.jpg"
    result = main(image_path)
    print(result)