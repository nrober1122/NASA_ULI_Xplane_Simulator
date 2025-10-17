import os
import pathlib
from typing import Optional
import ipdb
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
DT = config["DT"]

# def _load_network():
#     global _network

#     if _network is not None:
#         return

#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")
#     _network = TinyTaxiNetDNN()

#     # NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])
#     # scratch_dir = NASA_ULI_ROOT_DIR / "scratch"
#     # model_dir = NASA_ULI_ROOT_DIR / "models/tiny_taxinet_pytorch/morning/best_model.pt"
#     NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
#     model_dir = NASA_ULI_ROOT_DIR + '/models/tiny_taxinet_pytorch/morning/'
#     debug_dir = NASA_ULI_ROOT_DIR + '/scratch/debug/'
#     # assert model_dir.exists()


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
#         # _network = torch2jax(_network)
#         _network = torch_to_jax_model(_network)


# def get_network():
#     _load_network()
#     return _network

_network_loaded = False

# ---------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------

def _load_network_once():
    """Load the network into global scope exactly once."""
    global _network, _network_loaded
    if _network_loaded:
        return

    device = torch.device("cpu")
    model = TinyTaxiNetDNN()

    NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]
    model_dir = os.path.join(NASA_ULI_ROOT_DIR, "models/tiny_taxinet_pytorch/morning")
    best_model = os.path.join(model_dir, "best_model.pt")

    if USING_TORCH:
        state_dict = torch.load(best_model, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        _network = model
    else:
        # Load PyTorch weights, then convert to JAX once
        state_dict = torch.load(best_model, map_location=device)
        model.load_state_dict(state_dict)
        jax.config.update("jax_platform_name", "cpu")
        _network = torch_to_jax_model(model)

    _network_loaded = True


# def _load_network():
#     """Backwards-compatible alias for existing code."""
#     _load_network_once()


def get_network():
    """Public accessor."""
    _load_network_once()
    return _network


def evaluate_network(image: np.ndarray):
    _load_network_once()
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


def dynamics_sin(x, y, theta, tan_phi_rad, dt=0.05, v=5, L=5):
    """Dubin's car dynamics model (returns next state)

    Args:
        x: current crosstrack error (meters)
        y: current downtrack position (meters)
        theta: current heading error (degrees)
        phi_deg: steering angle input (degrees)
        -------------------------------
        dt: time step (seconds)
        v: speed (m/s)
        L: distance between front and back wheels (meters)
    """

    theta_rad = jnp.deg2rad(theta)

    x_dot = v * jnp.sin(theta_rad)
    theta_dot = (v / L) * tan_phi_rad

    x_prime = x + x_dot * dt
    theta_prime = theta + jnp.rad2deg(theta_dot) * dt

    return jnp.array([x_prime, theta_prime])


def dynamics(x, y, theta, tan_phi_rad, dt=0.05, v=5, L=5):
    """Dubin's car dynamics model (returns next state)

    Args:
        x: current crosstrack error (meters)
        y: current downtrack position (meters)
        theta: current heading error (degrees)
        phi_deg: steering angle input (degrees)
        -------------------------------
        dt: time step (seconds)
        v: speed (m/s)
        L: distance between front and back wheels (meters)
    """

    theta_rad = jnp.deg2rad(theta)
    theta2 = theta_rad * theta_rad
    theta3 = theta2 * theta_rad
    theta5 = theta3 * theta2
    theta7 = theta5 * theta2
    # phi_rad = jnp.deg2rad(phi_deg)

    # x_dot = v * jnp.sin(theta_rad)
    # y_dot = v * jnp.cos(theta_rad)
    # theta_dot = (v / L) * jnp.tan(phi_rad)

    # x_dot = v * jnp.sin(theta_rad)
    x_dot = v * (theta_rad - theta3 / 6 + theta5 / 120 - theta7 / 5040)  # Taylor approx for small angles
    # x_dot = v * theta_rad
    # x_dot = v * 1.1*jnp.tanh(theta_rad*0.99)
    # x_dot = v * 2.2*jax.nn.sigmoid(2*theta_rad) - 1.1
    theta_dot = (v / L) * tan_phi_rad

    x_prime = x + x_dot * dt
    theta_prime = theta + jnp.rad2deg(theta_dot) * dt

    return jnp.array([x_prime, theta_prime])


def evaluate_network_smoothed(observation: jnp.ndarray, alpha=0.7):
    """Evaluate the network with a simple exponential smoothing on the input observation.

    Args:
        observation (jnp.ndarray): The input observation to evaluate.
        alpha (float, optional): The smoothing factor. Defaults to 0.7.

    Returns:
        Tuple[float, float]: The predicted cte and heading.
    """
    _load_network_once()
    assert USING_TORCH is False, "Smoothing only implemented for JAX version"
    assert observation.shape == (131,)

    # prev_x_hat = observation[:2]
    prev_cte = observation[0]
    prev_he = observation[1]
    prev_phi = observation[2]
    image = observation[3:]
    # import ipdb; ipdb.set_trace()

    # print("Prev x_hat, theta_hat, phi:", prev_x_hat, , prev_phi.shape)

    state_hat_nn = _network(image).squeeze()
    state_hat_dyn = dynamics(
        prev_cte,
        0.0,
        prev_he,
        prev_phi,
        dt=DT,
    )
    state_hat = alpha * state_hat_nn + (1-alpha) * state_hat_dyn
    return state_hat


def package_input(x_prev, u_prev, image):
    """Package the observation into a single array for the network.

    Args:
        x_prev (float): Previous crosstrack error.
        u_prev (float): Previous heading error.
        image (np.ndarray): The input image.
    
    Returns:
        jnp.ndarray: The packaged observation.
    """
    assert image.shape == (128,)
    return jnp.concatenate([x_prev, u_prev, image])


def target_function(cte, he, max_rudder_deg=7):
    # return 2 - max_rudder_deg/10 * (cte - 10)
    return max_rudder_deg
