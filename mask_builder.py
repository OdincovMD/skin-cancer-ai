import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from resizeimage import resizeimage
from ultralytics import YOLO
import cv2

class BlockBuilder:

    """
    This class provides methods for creating building blocks of the encoder-decoder architecture, 
    including convolutional layers and pooling layers.
    """

    @staticmethod
    def create_enc_dec_block(in_dim: int, out_dim: int, is_last: bool=False) -> nn.Sequential:
        block = []
        block.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=out_dim),
                nn.ReLU()
            )
        )
        if is_last:
            block.append(
                nn.Conv2d(in_channels=out_dim, out_channels=1,
                          kernel_size=3, padding=1)
            )
        else:
            block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=out_dim,
                              out_channels=out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=out_dim),
                    nn.ReLU()
                )
            )

        return nn.Sequential(*block)

    @staticmethod
    def create_pool_block(is_unpool: bool=False, return_indices: bool=True) -> nn.MaxPool2d :
        if is_unpool:
            block = nn.MaxUnpool2d(2, stride=2)
        else:
            block = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        return block

class UNet(nn.Module):

    """
    The `UNet` class represents a convolutional neural network designed for segmentation tasks. 
    The network consists of an encoder, a decoder, and a "bottleneck" section for compressing information.
    """

    def __init__(self):
        super().__init__()

        builder = BlockBuilder()

        self.enc_conv0 = builder.create_enc_dec_block(3, 64)
        self.pool0 = builder.create_pool_block(return_indices=False)
        self.enc_conv1 = builder.create_enc_dec_block(64, 128)
        self.pool1 = builder.create_pool_block(return_indices=False)
        self.enc_conv2 = builder.create_enc_dec_block(128, 256)
        self.pool2 = builder.create_pool_block(return_indices=False)
        self.enc_conv3 = builder.create_enc_dec_block(256, 512)
        self.pool3 = builder.create_pool_block(return_indices=False)

        self.bottleneck_conv = builder.create_enc_dec_block(512, 1024)

        self.upsample0 = nn.Upsample(32)
        self.dec_conv0 = builder.create_enc_dec_block(1024+512, 512)
        self.upsample1 = nn.Upsample(64)
        self.dec_conv1 = builder.create_enc_dec_block(512+256, 256)
        self.upsample2 = nn.Upsample(128)
        self.dec_conv2 = builder.create_enc_dec_block(256+128, 128)
        self.upsample3 = nn.Upsample(256)
        self.dec_conv3 = builder.create_enc_dec_block(128+64, 32, True)

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(self.pool0(e0))
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))

        # bottleneck
        b = self.bottleneck_conv(self.pool3(e3))

        # decoder
        d0 = self.upsample0(b)
        d0 = torch.cat([d0, e3], dim=1)
        d0 = self.dec_conv0(d0)

        d1 = self.upsample1(d0)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec_conv1(d1)

        d2 = self.upsample2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec_conv2(d2)

        d3 = self.upsample3(d2)
        d3 = torch.cat([d3, e0], dim=1)
        d3 = self.dec_conv3(d3)  # no activation
        return d3

def get_prediction(image_path: str) -> np.ndarray:
    """
    This function loads a pre-trained UNet model, makes a segmentation mask prediction 
    for a single image, and returns the prediction as an array.

    Parameters
    ----------
        image_path: str
            Path to the image file.
        device: str
            The device for computation ('cuda:0' for GPU, 'cpu' for CPU). Defaults to GPU if available.

    Returns
    -------
        mask: np.ndarray
            The predicted segmentation mask as a binary array (0 or 1).
    """
    
    image = Image.open(image_path)

    image = np.array(resizeimage.resize_cover(image, [256, 256], validate=False))
    image = np.rollaxis(image, 2, 0)  # Move channels

    transformed_image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    
    model = UNet()
    model.load_state_dict(
        torch.load("weight/mask_builder_unet.pth", weights_only=True, map_location=torch.device('cpu'))
        )
    model.eval()
    
    with torch.no_grad():
        output = model(transformed_image)
        pred = torch.sigmoid(output)
        pred = pred > 0.5  # Threshold to create a binary mask
        pred = pred.cpu().numpy()[0, 0]  # Convert to numpy, get the first channel
    
    return pred

def main(path_to_image: str)-> np.ndarray:
    """
    This function loads an image from the specified path, prepares it for the model, 
    makes a segmentation mask prediction, and returns the mask resized to the original image dimensions.

    Parameters
    ----------
    path_to_image : str
            The path to the image for which a mask prediction needs to be made.

    Returns
    -------
    mask : np.ndarray
        The segmentation mask resized to the original image dimensions, in the form of a NumPy array.
    """ 

    model = YOLO("weight/v2310.pt")

    results = model(path_to_image, retina_masks=True, save=True)

    orig_img = results[0].orig_img
    height, width = orig_img.shape[:2]

    combined_mask = np.zeros((height, width), dtype=np.uint8)

    if results[0].masks and results[0].masks.xy: 
        for mask in results[0].masks.xy:
            mask_points = np.array(mask, dtype=np.int32)
            instance_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(instance_mask, [mask_points], 255)
            combined_mask = cv2.bitwise_or(combined_mask, instance_mask)
    else:
        mask = get_prediction(path_to_image)
        mask = Image.fromarray(mask)
        mask_resized = mask.resize((width, height), resample=Image.NEAREST)
        combined_mask = np.array(mask_resized, dtype=np.uint8)
    return combined_mask
