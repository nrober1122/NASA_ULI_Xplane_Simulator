import os
import time
from typing import Optional
import yaml

import cv2
import mss
import numpy as np
import torch
import jax
import jax.numpy as jnp
from simulators.NASA_ULI_Xplane_Simulator.src.simulation.nnet import *
from PIL import Image
from torchvision import transforms

# from utils.torch2jax import torch2jax
from utils.torch2jaxmodel import torch_to_jax_model

from simulators.NASA_ULI_Xplane_Simulator.src.train_DNN.model_taxinet import TaxiNetDNN, TaxiNetCNN
from simulators.NASA_ULI_Xplane_Simulator.src.simulation.tiny_taxinet2 import dynamics

_network: Optional[TaxiNetDNN] = None

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

USING_TORCH = config["USING_TORCH"]
DT = config["DT"]

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
# width = 224    # Width of image
# height = 224    # Height of image

screenShot = mss.mss()
monitor = {'top': 100, 'left': 100, 'width': 1720, 'height': 960}
screen_width = 360  # For cropping
screen_height = 200  # For cropping

device = torch.device("cpu")
print('found device: ', device)



_network_loaded = False

# ---------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------

def _load_network_once(in_channels=3, H=224, W=224):
    """Load the network into global scope exactly once."""
    global _network, _network_loaded
    if _network_loaded:
        return

    torch.cuda.empty_cache()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
    if config["STATE_ESTIMATOR"] == "dnn":
        _network = TaxiNetDNN()
        model_dir = NASA_ULI_ROOT_DIR + '/models/pretrained_DNN_nick/'
    elif config["STATE_ESTIMATOR"] == "cnn":
        _network = TaxiNetCNN(input_channels=in_channels, H=H, W=W)
        model_dir = NASA_ULI_ROOT_DIR + '/models/cnn_taxinet/'
    elif config["STATE_ESTIMATOR"] == "cnn64":
        _network = TaxiNetCNN(input_channels=in_channels, H=H, W=W)
        model_dir = NASA_ULI_ROOT_DIR + '/models/cnn64_taxinet/'
    else:
        raise ValueError("Invalid STATE_ESTIMATOR")

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
        _network = torch_to_jax_model(_network)

    _network_loaded = True


# def _load_network():
#     """Backwards-compatible alias for existing code."""
#     _load_network_once()


def get_network(in_channels=3, H=224, W=224):
    """Public accessor."""
    _load_network_once(in_channels=in_channels, H=H, W=W)
    return _network


# def _load_network(in_channels=3, H=224, W=224):
#     global _network

#     if _network is not None:
#         return _network
    
#     torch.cuda.empty_cache()
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")
#     print('found device: ', device)

#     # Read in the network
#     NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']

#     if config["STATE_ESTIMATOR"] == "dnn":
#         _network = TaxiNetDNN()
#         model_dir = NASA_ULI_ROOT_DIR + '/models/pretrained_DNN_nick/'
#     elif config["STATE_ESTIMATOR"] == "cnn":
#         _network = TaxiNetCNN(input_channels=in_channels, H=H, W=W)
#         model_dir = NASA_ULI_ROOT_DIR + '/models/cnn_taxinet/'
#     elif config["STATE_ESTIMATOR"] == "cnn64":
#         _network = TaxiNetCNN(input_channels=in_channels, H=H, W=W)
#         model_dir = NASA_ULI_ROOT_DIR + '/models/cnn64_taxinet/'
#     else:
#         raise ValueError("Invalid STATE_ESTIMATOR")

#     # load the pre-trained model
#     if USING_TORCH:
#         if device.type == 'cpu':
#             _network.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
#         else:
#             _network.load_state_dict(torch.load(model_dir + '/best_model.pt'))

#         _network = _network.to(device)
#         _network.eval()
#     else:
#         _network.load_state_dict(torch.load(model_dir + '/best_model.pt', map_location=torch.device('cpu')))
#         # import ipdb; ipdb.set_trace()
#         jax.config.update("jax_platform_name", "cpu")
#         _network = torch_to_jax_model(_network)


# def get_network(in_channels=3, H=224, W=224):
#     _load_network(in_channels, H, W)
#     # _network.model.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
#     return _network


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


def _normalize_image(image: np.ndarray, in_channels: int, width: int, height: int) -> torch.Tensor:
    """Normalize the image, then flattens it. Input: (8, 16). Output: (128,)."""
    pil_image = Image.fromarray(image)
    assert pil_image.size == (screen_width, screen_height)

    if in_channels == 3:
        tfms = transforms.Compose([transforms.Resize((height, width)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225]),])
    elif in_channels == 1:
        mean, std = [0.5], [0.5]
        tfms = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    return tfms(pil_image)


def process_image(image: np.ndarray, in_channels: int, width: int, height: int) -> np.ndarray:
    """Process the image for use with TinyTaxiNet.
    Input: (1080, 1920, 4)
    Output: (1, 3, 224, 224)
    """
    assert image.shape == (1080, 1920, 4)

    image_processed = _normalize_image(_crop_image(image), in_channels, width, height)

    assert image_processed.shape == (in_channels, width, height)
    return image_processed


def evaluate_network(image: np.ndarray, in_channels: int, width: int, height: int) -> np.ndarray:
    """Evaluate the network on the preprocessed image.
    Image: (128 = 8 * 16,)
    """
    _load_network_once(in_channels=in_channels, H=height, W=width)
    assert image.shape == (in_channels, width, height)

    image = image.reshape(1, in_channels, width, height)
    # print(torch.version.cuda)        # Should match your installed CUDA toolkit
    # print(torch.backends.cudnn.version())  # Should be a valid cuDNN version
    # print(torch.cuda.is_available())       # Should be True

    if USING_TORCH:
        image = image.to(device)
    else:
        # Convert to JAX
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            image = jax.numpy.array(image)

    pred = _network(image)

    if USING_TORCH:
        pred = pred.cpu().detach().numpy().flatten()
    else:
        pred = jax.device_get(pred).squeeze()

    # return scaled output
    # return pred[0]*10, pred[1]*30
    return pred*jnp.array([10.0, 30.0])


def evaluate_network_smoothed(image: np.ndarray, in_channels: int, width: int, height: int) -> np.ndarray:
    """Evaluate the network on the preprocessed image.
    Image: (128 = 8 * 16,)
    """
    _load_network_once(in_channels=in_channels, H=height, W=width)
    assert image.shape == (in_channels, width, height)

    image = image.reshape(1, in_channels, width, height)
    # print(torch.version.cuda)        # Should match your installed CUDA toolkit
    # print(torch.backends.cudnn.version())  # Should be a valid cuDNN version
    # print(torch.cuda.is_available())       # Should be True

    if USING_TORCH:
        image = image.to(device)
    else:
        # Convert to JAX
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            image = jax.numpy.array(image)

    pred = _network(image)

    if USING_TORCH:
        pred = pred.cpu().detach().numpy().flatten()
    else:
        pred = jax.device_get(pred).squeeze()

    # return scaled output
    # return pred[0]*10, pred[1]*30
    return pred*jnp.array([10.0, 30.0])

def evaluate_network_smoothed(observation: jnp.ndarray,
                              in_channels: int,
                              width: int,
                              height: int,
                              alpha: float = 0.7
                              ) -> np.ndarray:
    
    """Evaluate the network with a simple exponential smoothing on the input observation.

    Args:
        observation (jnp.ndarray): The input observation to evaluate.
        alpha (float, optional): The smoothing factor. Defaults to 0.7.

    Returns:
        Tuple[float, float]: The predicted cte and heading.
    """
    
    assert observation.shape == (in_channels + 1, width, height)
    _load_network_once(
        in_channels=in_channels,
        H=height,
        W=width
    )
    assert USING_TORCH is False, "Smoothing only implemented for JAX version"

    # prev_x_hat = observation[:2]
    prev_cte = observation[0, 0, 0]
    prev_he = observation[0, 0, 1]
    prev_phi = observation[0, 0, 2]
    image = observation[1:]
    # import ipdb; ipdb.set_trace()

    # print("Prev x_hat, theta_hat, phi:", prev_x_hat, , prev_phi.shape)

    state_hat_nn = evaluate_network(image, in_channels, width, height)
    state_hat_dyn = dynamics(
        prev_cte,
        0.0,
        prev_he,
        prev_phi,
        dt=DT,
    )
    state_hat = alpha * state_hat_nn + (1-alpha) * state_hat_dyn
    return state_hat


def target_function(cte, he,  max_rudder_deg=7):
    """Given current cte and he, return the target rudder angle."""
    return 7
    # return 6 - 7/10 * (cte - 10)


def package_input(x_prev, u_prev, image):
    """Package the input for the network when using smoothing.
    x_prev: (2,) = (cte, he)
    u_prev: (1,) = (rudder angle)
    image_processed: (in_channels, width, height)
    Output: (2 + 1 + in_channels * width * height,)
    """
    assert x_prev.shape == (2,)
    assert u_prev.shape == (1,)

    dyn_terms = jnp.zeros((1, image.shape[1], image.shape[2]))  # (batch, features, state(4) + input(2))
    dyn_terms = dyn_terms.at[0, 0, 0:2].set(x_prev)
    dyn_terms = dyn_terms.at[0, 0, [2]].set(u_prev)
    observation = jnp.concatenate([dyn_terms, image], axis=0)

    return observation

# def evaluate_network_jax(image: np.ndarray, in_channels: int, width: int, height: int) -> jnp.ndarray:
#     """Evaluate the network on the preprocessed image.
#     Image: (128 = 8 * 16,)
#     """
#     _load_network(in_channels, width, height)
#     assert image.shape == (in_channels, width, height)

#     image = image.reshape(1, in_channels, width, height)
#     # print(torch.version.cuda)        # Should match your installed CUDA toolkit
#     # print(torch.backends.cudnn.version())  # Should be a valid cuDNN version
#     # print(torch.cuda.is_available())       # Should be True

#     if USING_TORCH:
#         image = image.to(device)
#     else:
#         # Convert to JAX
#         if isinstance(image, torch.Tensor):
#             image = image.detach().cpu().numpy()
#             image = jax.numpy.array(image)

#     pred = _network(image)

#     if USING_TORCH:
#         pred = pred.cpu().detach().numpy().flatten()
#         pred = jax.numpy.array(pred).squeeze()
#     else:
#         pred = jax.device_get(pred).squeeze()

#     # return scaled output
#     # return pred[0]*10, pred[1]*30
#     return pred*jnp.array([10.0, 30.0])
