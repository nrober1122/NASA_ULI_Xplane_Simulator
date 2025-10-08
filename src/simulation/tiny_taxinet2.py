import os
import pathlib
from typing import Optional
import yaml

import numpy as np
import torch
import jax
import jax.numpy as jnp

# from utils.torch2jax import torch2jax
from utils.torch2jaxmodel import torch_to_jax_model

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# from tiny_taxinet_train.model_tiny_taxinet import TinyTaxiNetDNN
from simulators.NASA_ULI_Xplane_Simulator.src.tiny_taxinet_train.model_tiny_taxinet import TinyTaxiNetDNN

_network: Optional[TinyTaxiNetDNN] = None
USING_TORCH = config["USING_TORCH"]

def _load_network():
    global _network

    if _network is not None:
        return

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    _network = TinyTaxiNetDNN()

    # NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])
    # scratch_dir = NASA_ULI_ROOT_DIR / "scratch"
    # model_dir = NASA_ULI_ROOT_DIR / "models/tiny_taxinet_pytorch/morning/best_model.pt"
    NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
    model_dir = NASA_ULI_ROOT_DIR + '/models/tiny_taxinet_pytorch/morning/'
    debug_dir = NASA_ULI_ROOT_DIR + '/scratch/debug/'
    # assert model_dir.exists()


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
        # _network = torch2jax(_network)
        _network = torch_to_jax_model(_network)


def get_network():
    _load_network()
    return _network


def evaluate_network(image: np.ndarray):
    _load_network()
    assert image.shape == (128,)
    # Unlike the nnet one, this takes does the flattening inside. We can still flatten though and it should be fine.
    if USING_TORCH:
        with torch.inference_mode():
            pred = _network.forward(torch.Tensor(image[None, :, None]))
            pred = pred.numpy().squeeze()
            # [ cte, heading ]
            return pred[0], pred[1]
    else:
        # Convert to JAX
        # image = jax.numpy.array(image).reshape(-1, 1)
        pred = _network(image).squeeze()
        # pred = jax.device_get(pred).squeeze()
        # [ cte, heading ]
        return pred
