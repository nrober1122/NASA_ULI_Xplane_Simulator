import os
import pathlib
import time
from typing import Optional

import cv2
import mss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tiny_taxinet_train.model_tiny_taxinet import TinyTaxiNetDNN

network: Optional[TinyTaxiNetDNN] = None


def _load_network():
    global network

    if network is not None:
        return

    NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])
    scratch_dir = NASA_ULI_ROOT_DIR / "scratch"
    model_path = scratch_dir / "tiny_taxinet_DNN_train/morning/best_model.pt"
    assert model_path.exists()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = TinyTaxiNetDNN()
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()


def evaluate_network(image: np.ndarray):
    _load_network()
    assert image.shape == (128,)
    # Unlike the nnet one, this takes does the flattening inside. We can still flatten though and it should be fine.
    with torch.inference_mode():
        pred = network.forward(torch.Tensor(image[None, :, None]))
        pred = pred.numpy().squeeze()
        return pred[0], pred[1]
