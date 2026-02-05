import os
import time

import cv2
import einops as ei
import mss
import numpy as np
from loguru import logger
from simulators.NASA_ULI_Xplane_Simulator.src.simulation.nnet import *
from PIL import Image
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# Read in the network

filename = THIS_DIR / "../../models/TinyTaxiNet.nnet"
network = NNet(filename)

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
stride = 16  # Size of square of pixels downsampled to one grayscale value
# During downsampling, average the numPix brightest pixels in each square
numPix = 16
width = 256 // stride  # Width of downsampled grayscale image
height = 128 // stride  # Height of downsampled grayscale image

screenShot = mss.mss()
monitor = {"top": 100, "left": 100, "width": 1720, "height": 960}
screen_width = 360  # For cropping
screen_height = 200  # For cropping


def getCurrentImage():
    """Returns a downsampled image of the current X-Plane 11 image
    compatible with the TinyTaxiNet neural network state estimator

    NOTE: this is designed for screens with 1920x1080 resolution
    operating X-Plane 11 in full screen mode - it will need to be adjusted
    for other resolutions
    """
    # Get current screenshot
    # (960, 1720, 4)
    img = np.array(screenShot.grab(monitor))
    # logger.info("1: img.shape: {}".format(img.shape))

    # (730, 1720, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)[230:, :, :]
    # logger.info("2: img.shape: {}".format(img.shape))

    # (200, 360, 3)
    img = cv2.resize(img, (screen_width, screen_height))
    # logger.info("3: img.shape: {}".format(img.shape))
    img = img[:, :, ::-1]
    img = np.array(img)

    # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so
    # values range between 0 and 1
    img = np.array(Image.fromarray(img).convert("L").crop((55, 5, 360, 135)).resize((256, 128)))
    # Image.fromarray(img).save("taxinet_img.png")
    img = img / 255.0

    # Downsample image
    # Split image into stride x stride boxes, average numPix brightest pixels in that box
    # As a result, img2 has one value for every box
    img2 = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            img2[i, j] = np.mean(
                np.sort(img[stride * i : stride * (i + 1), stride * j : stride * (j + 1)].reshape(-1))[-numPix:]
            )

    # Image.fromarray((img2 * 255).astype(np.uint8)).save("taxinet_img_downsampled.png")

    # Ensure that the mean of the image is 0.5 and that values range between 0 and 1
    # The training data only contains images from sunny, 9am conditions.
    # Biasing the image helps the network generalize to different lighting conditions (cloudy, noon, etc)
    img2 -= img2.mean()
    img2 += 0.5
    img2[img2 > 1] = 1
    img2[img2 < 0] = 0
    return img2.flatten()


def getStateTinyTaxiNet(client):
    """Returns an estimate of the crosstrack error (meters)
    and heading error (degrees) by passing the current
    image through TinyTaxiNet

    Args:
        client: XPlane Client
    """
    image = getCurrentImage()
    pred = network.evaluate_network(image)
    return pred[0], pred[1]


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

    # 5: Convert to grayscale.
    image = Image.fromarray(image).convert("L")

    # 6: Crop out nose, sky, bottom of image.
    image = image.crop((55, 5, 360, 135)).resize((256, 128))
    image = np.array(image) / 255.0
    assert image.shape == (128, 256)

    return image


def _downsample_image(image: np.ndarray):
    assert image.shape == (128, 256)
    height_orig, width_orig = image.shape

    stride = 16
    num_pix = 16
    width = width_orig // stride
    height = height_orig // stride

    # Downsample by taking the mean of the num_pix brightest pixels in each stride x stride box.
    patches = ei.rearrange(image, "(nh sh) (nw sw) -> nh nw sh sw", nw=width, nh=height, sw=stride, sh=stride)
    patches_flat = ei.rearrange(patches, "nh nw sh sw -> nh nw (sh sw)")
    out = np.mean(np.sort(patches_flat, axis=-1)[:, :, -num_pix:], axis=-1)
    return out


def _normalize_image(image: np.ndarray):
    """Normalize the image, then flattens it. Input: (8, 16). Output: (128,)."""
    assert image.shape == (8, 16)
    image -= image.mean()
    image += 0.5
    image[image > 1] = 1
    image[image < 0] = 0
    return image.flatten()


def process_image(image: np.ndarray) -> np.ndarray:
    """Process the image for use with TinyTaxiNet.
    Input: (1080, 1920, 4)
    Output: (128 = 8 * 16,)
    """
    assert image.shape == (1080, 1920, 4)
    image = _normalize_image(_downsample_image(_crop_image(image)))
    image = image.clip(0, 1)
    assert image.shape == (128,)
    return image


def evaluate_network(image: np.ndarray):
    """Evaluate the network on the preprocessed image.
    Image: (128 = 8 * 16,)
    """
    pred = network.evaluate_network(image)
    return pred[0], pred[1]
