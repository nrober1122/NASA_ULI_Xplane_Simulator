import os
import pathlib
from typing import Optional

import numpy as np
import torch

from attack.train_static_tinytaxinet import TinyTaxiNetAttackStatic

_network: Optional[TinyTaxiNetAttackStatic] = None


def _load_network():
    global _network

    if _network is not None:
        return

    NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])
    models_dir = NASA_ULI_ROOT_DIR / "models"
    # scratch_dir = NASA_ULI_ROOT_DIR / "scratch"
    model_path = models_dir / "tiny_taxinet_attack_static/model_atk.pt"
    assert model_path.exists()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = (8, 16)
    _network = TinyTaxiNetAttackStatic(image_size, 0.0, 0.0)
    _network.load_state_dict(torch.load(model_path, map_location=device))
    _network.eval()


def get_network():
    _load_network()
    return _network


# def get_patch(image: np.ndarray):
#     _load_network()
#     assert image.shape == (128,)
#     with torch.inference_mode():
#         return _network.patch.detach().cpu().numpy().flatten()

def get_patch(image: np.ndarray, linfnorm: float = 0.027):
    _load_network()
    assert image.shape == (128,)
    with torch.inference_mode():
        atk = _network.patch.detach().cpu().numpy().flatten()
        return (linfnorm*atk/np.abs(np.min(atk))).clip(-linfnorm, linfnorm)

