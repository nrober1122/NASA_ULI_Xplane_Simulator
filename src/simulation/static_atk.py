import os
import pathlib
from typing import Optional

import numpy as np
import torch

from attack.train_static_tinytaxinet import TinyTaxiNetAttackStatic


class StaticAttack:
    def __init__(self, name: str):
        self.name = name
        self._network: Optional[TinyTaxiNetAttackStatic] = None

    def _get_model_path(self):
        NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])
        models_dir = NASA_ULI_ROOT_DIR / "models"
        scratch_dir = NASA_ULI_ROOT_DIR / "scratch"

        if self.name == "static_rudder":
            # return models_dir / "tiny_taxinet_attack_static/model_atk.pt"
            return scratch_dir / "tiny_taxinet_attack_static_mse/model_atk.pt"
        if self.name == "lyap":
            return scratch_dir / "tiny_taxinet_attack_static_lyap/model_atk.pt"

        raise NotImplementedError(f"Unknown attack name: {self.name}")

    def _load_network(self):
        if self._network is not None:
            return

        model_path = self._get_model_path()
        assert model_path.exists()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image_size = (8, 16)
        self._network = TinyTaxiNetAttackStatic(image_size, 0.0, 0.0)
        self._network.load_state_dict(torch.load(model_path, map_location=device))
        self._network.eval()

    def get_network(self):
        self._load_network()
        return self._network

    # def get_patch(image: np.ndarray):
    #     _load_network()
    #     assert image.shape == (128,)
    #     with torch.inference_mode():
    #         return _network.patch.detach().cpu().numpy().flatten()

    def get_patch(self, image: np.ndarray, linfnorm: float = 0.027):
        self._load_network()
        assert image.shape == (128,)
        with torch.inference_mode():
            # Allow scaling of the learned patch
            atk = self._network.patch.detach().cpu().numpy().flatten()
            # Scale so that abs(norm_patch) is at most 1.
            norm_patch = atk / float(self._network.max_delta.detach().cpu())
            atk = norm_patch * linfnorm
            # norm_patch = atk / 0.027
            # return linfnorm * norm_patch
            return atk
