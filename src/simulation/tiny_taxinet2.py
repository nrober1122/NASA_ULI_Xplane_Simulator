import os
import pathlib
from typing import Optional

import numpy as np
import torch

from tiny_taxinet_train.model_tiny_taxinet import TinyTaxiNetDNN

_network: Optional[TinyTaxiNetDNN] = None

NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])


def _load_network():
    global _network

    if _network is not None:
        return

    scratch_dir = NASA_ULI_ROOT_DIR / "scratch"
    model_path = NASA_ULI_ROOT_DIR / "models/tiny_taxinet_pytorch/morning/best_model.pt"
    assert model_path.exists()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _network = TinyTaxiNetDNN()
    _network.load_state_dict(torch.load(model_path, map_location=device))
    _network.eval()


def get_network():
    _load_network()
    return _network


def get_stride_network(stride: int):
    models_dir = NASA_ULI_ROOT_DIR / "models"
    model_path = models_dir / "tiny_taxinet_torch_stride/tiny_taxinet_train_stride{}/best_model.pt".format(stride)
    assert model_path.exists()

    width = 256 // stride
    height = 128 // stride
    n_features_in = width * height

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = TinyTaxiNetDNN(n_features_in=n_features_in)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()
    return _network


def evaluate_network(image: np.ndarray):
    _load_network()
    assert image.shape == (128,)
    # Unlike the nnet one, this takes does the flattening inside. We can still flatten though and it should be fine.
    with torch.inference_mode():
        pred = _network.forward(torch.Tensor(image[None, :, None]))
        pred = pred.numpy().squeeze()
        # [ cte, heading ]
        return pred[0], pred[1]
