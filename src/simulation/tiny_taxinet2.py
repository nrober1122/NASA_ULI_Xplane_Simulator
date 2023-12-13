import os
import pathlib
from typing import Optional

import numpy as np
import torch
from loguru import logger

from tiny_taxinet_train.model_tiny_taxinet import TinyTaxiNetDNN

NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])

_cache = {}


class StateEstimator:
    def __init__(self, stride: int = 16):
        self.stride = stride
        self._network: Optional[TinyTaxiNetDNN] = None

    def _load_network(self):

        if self._network is not None:
            return

        scratch_dir = NASA_ULI_ROOT_DIR / "scratch"
        models_dir = NASA_ULI_ROOT_DIR / "models"

        if self.stride == 16 or self.stride is None:
            model_path = models_dir / "tiny_taxinet_pytorch/morning/best_model.pt"
        else:
            model_path = models_dir / "tiny_taxinet_torch_stride/tiny_taxinet_train_stride{}/best_model.pt".format(
                self.stride
            )
        assert model_path.exists()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global _cache
        if self.stride in _cache:
            logger.info(f"Loading stride {self.stride} from cache!")
            self._network = _cache[self.stride].to(device)

        width = 256 // self.stride
        height = 128 // self.stride
        n_features_in = width * height

        self._network = TinyTaxiNetDNN(n_features_in=n_features_in)
        self._network.load_state_dict(torch.load(model_path, map_location=device))
        self._network.eval()

        if self.stride not in _cache:
            _cache[self.stride] = self._network.cpu()

    def get_network(self):
        self._load_network()
        return self._network

    def evaluate_network(self, image: np.ndarray):
        self._load_network()
        # Unlike the nnet one, this takes does the flattening inside. We can still flatten though and it should be fine.
        with torch.inference_mode():
            pred = self._network.forward(torch.Tensor(image[None, :, None]))
            pred = pred.numpy().squeeze()
            # [ cte, heading ]
            return pred[0], pred[1]

    def __call__(self, image: np.ndarray):
        return self.evaluate_network(image)


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
    return network
