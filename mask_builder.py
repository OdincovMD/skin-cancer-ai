import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from resizeimage import resizeimage
from PIL import Image


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
    
class ImageFileDataset(Dataset):

    """
    The `ImageFileDataset` class is designed to work with a single image loaded from disk. It can be used for testing models on individual images.

    Parameters
    ----------
        file_path: str
            The path to the image file.
        transform
            The transformations that will be applied to the image.

    """

    def __init__(self, file_path: str, transform):
        self.file_path = file_path
        self.transform = transform
        # Проверка на расширение файла
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset contains only one image.")
        
        image = Image.open(self.file_path)
        if self.transform:
            image = self.transform(image)
        
        return image, self.file_path  # Возвращаем путь для использования при сохранении предсказаний


def get_predictions(dataloader: DataLoader, device: str='cuda:0') -> np.ndarray:

    """
    This function loads a pre-trained UNet model, makes segmentation mask predictions for images loaded through a dataloader, 
    and returns the predictions for the first image as an array.

    Parameters
    ----------
        dataloader: DataLoader
            A DataLoader object containing images for prediction.
            Each element of the dataloader is a tuple consisting of the image and its path.
        device: str=None
            The device on which the computation will be performed ('cuda:0' for using GPU, 'cpu' for CPU). 
            By default, GPU is used if available.

    Returns
    -------
        mask: np.ndarray
            The predicted mask for the first image in the dataloader as a binary array (0 or 1).
    """

    model = UNet().to(device)
    model.load_state_dict(torch.load("/home/hardbox/python/skin/weight/model_weights.pth", weights_only=True, map_location=torch.device(device)))
    model.eval()
    with torch.no_grad():
        for images, img_paths in dataloader:
            images = images.to(device, dtype=torch.float)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = preds > 0.7
            print(img_paths)
            preds = preds.cpu().numpy()
            for pred, _ in zip(preds, img_paths):
                return pred[0]

def main(path_to_image: str) -> np.ndarray:    

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def transform(image):
        image = np.array(resizeimage.resize_cover(image, [256, 256], validate=False))
        image = np.rollaxis(image, 2, 0)  # Перекладываем каналы
        return image
    
    dataset = ImageFileDataset(path_to_image, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    mask = get_predictions(data_loader, device=device)

    original = Image.open(path_to_image)
    mask = Image.fromarray(mask)
    mask_resized = mask.resize(original.size, resample=Image.NEAREST)
    mask = np.array(mask_resized, dtype=np.uint8)
    return mask

if __name__ == "__main__":
    main("26.jpg")
