import os
import time
from typing import Optional
import yaml

import cv2
import mss
import numpy as np
import torch
import jax
from nnet import *
from PIL import Image
from torchvision import transforms

from utils.torch2jax import torch2jax

from train_DNN.model_taxinet import TaxiNetDNN

_network: Optional[TaxiNetDNN] = None

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

USING_TORCH = config["USING_TORCH"]

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
width = 224    # Width of image
height = 224    # Height of image

screenShot = mss.mss()
monitor = {'top': 100, 'left': 100, 'width': 1720, 'height': 960}
screen_width = 360  # For cropping
screen_height = 200  # For cropping

device = torch.device("cpu")
print('found device: ', device)


def _load_network():
    global _network

    if _network is not None:
        return _network
    
    torch.cuda.empty_cache()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print('found device: ', device)

    _network = TaxiNetDNN()

    # Read in the network
    NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
    model_dir = NASA_ULI_ROOT_DIR + '/models/pretrained_DNN_nick/'

    # load the pre-trained model
    if USING_TORCH:
        if device.type == 'cpu':
            _network.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
        else:
            _network.load_state_dict(torch.load(model_dir + '/best_model.pt'))

        _network = _network.to(device)
        _network.eval()
    else:
        _network.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
        # import ipdb; ipdb.set_trace()
        jax.config.update("jax_platform_name", "cpu")
        _network = torch2jax(_network)


def get_network():
    _load_network()
    _network.model.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    return _network


def _crop_image(image_arr: np.ndarray):
    assert image_arr.shape == (1080, 1920, 4)
    # 0: Remove alpha channel.
    image = image_arr[:, :, :3]

    # 1: Crop 100 pixels from the top and bottom, 100 pixels from the left, 20 pixels from the right.
    image = image[100:-20, 100:-100, :]
    assert image.shape == (960, 1720, 3)

    # 2: Crop 230 more pixels from the top.
    image = image[230:, :, :]
    assert image.shape == (730, 1720, 3)

    # 3: Resize image to 360x200.
    image = cv2.resize(image, (360, 200))
    assert image.shape == (200, 360, 3)

    # 4: BGR -> RGB.
    image = image[:, :, ::-1]

    return image


def _normalize_image(image: np.ndarray):
    """Normalize the image, then flattens it. Input: (8, 16). Output: (128,)."""
    pil_image = Image.fromarray(image)
    assert pil_image.size == (screen_width, screen_height)
    tfms = transforms.Compose([transforms.Resize((width, height)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225]),])
    
    return tfms(pil_image)


def process_image(image: np.ndarray) -> np.ndarray:
    """Process the image for use with TinyTaxiNet.
    Input: (1080, 1920, 4)
    Output: (1, 3, 224, 224)
    """
    assert image.shape == (1080, 1920, 4)

    image_processed = _normalize_image(_crop_image(image))

    assert image_processed.shape == (3, width, height)
    return image_processed


def evaluate_network(image: np.ndarray):
    """Evaluate the network on the preprocessed image.
    Image: (128 = 8 * 16,)
    """
    _load_network()
    assert image.shape == (3, width, height)

    image = image.reshape(1, 3, width, height)
    # print(torch.version.cuda)        # Should match your installed CUDA toolkit
    # print(torch.backends.cudnn.version())  # Should be a valid cuDNN version
    # print(torch.cuda.is_available())       # Should be True

    if USING_TORCH:
        image = image.to(device)
    else:
        # Convert to JAX
        image = image.detach().cpu().numpy()
        image = jax.numpy.array(image)

    pred = _network(image)

    if USING_TORCH:
        pred = pred.cpu().detach().numpy().flatten()
    else:
        pred = jax.device_get(pred).squeeze()

    # return scaled output
    return pred[0]*10, pred[1]*30