import os
import pathlib
from typing import Optional

import numpy as np
import torch
from loguru import logger

from attack.train_static_tinytaxinet import TinyTaxiNetAttackStatic
from simulation.tiny_taxinet import process_image


class StaticAttack:
    def __init__(self, name: str = None, stride: int = None):
        self.name = name
        self._network: Optional[TinyTaxiNetAttackStatic] = None
        if stride == 16:
            stride = None
        self._stride = stride
        logger.info("Attacking with stride = {}!".format(stride))

    @property
    def stride(self) -> int:
        if self._stride is None:
            return 16
        return self._stride

    def _get_model_path(self):
        NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])
        models_dir = NASA_ULI_ROOT_DIR / "models"
        scratch_dir = NASA_ULI_ROOT_DIR / "scratch"

        if self._stride is None:
            if self.name == "static_rudder":
                # return models_dir / "tiny_taxinet_attack_static/model_atk.pt"
                return scratch_dir / "tiny_taxinet_attack_static_mse/model_atk.pt"
            if self.name == "lyap":
                return scratch_dir / "tiny_taxinet_attack_static_lyap/model_atk.pt"
        else:
            return scratch_dir / f"tiny_taxinet_attack_static_stride{self._stride}_mse/model_atk.pt"

        raise NotImplementedError(f"Unknown attack name: {self.name}")

    def _load_network(self):
        if self._network is not None:
            return

        model_path = self._get_model_path()
        assert model_path.exists()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        width = 256 // self.stride
        height = 128 // self.stride
        image_size = (height, width)
        self._network = TinyTaxiNetAttackStatic(image_size, 0.0, 0.0)
        self._network.load_state_dict(torch.load(model_path, map_location=device))
        self._network.eval()
        logger.info("Patch was trained with max_delta = {}".format(self._network.max_delta.detach().cpu()))

    def get_network(self):
        self._load_network()
        return self._network

    def get_patch(self, image: np.ndarray, linfnorm: float = 0.027):
        self._load_network()
        n_image_feats = np.prod(self._network.patch.shape)
        assert image.shape == (n_image_feats,)
        with torch.inference_mode():
            # Allow scaling of the learned patch
            atk = self._network.patch.detach().cpu().numpy().flatten()
            # Scale so that abs(norm_patch) is at most 1.
            norm_patch = atk / float(self._network.max_delta.detach().cpu())
            atk = norm_patch * linfnorm
            # norm_patch = atk / 0.027
            # return linfnorm * norm_patch
            return atk

    def process_image(self, image: np.ndarray):
        return process_image(image, stride=self._stride)
