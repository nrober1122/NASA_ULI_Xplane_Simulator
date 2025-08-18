import os
import sys
from datetime import datetime

xpc3_dir = os.environ["NASA_ULI_ROOT_DIR"] + "/src/"

sys.path.append(xpc3_dir)

import time
from typing import Callable

import ipdb
import time
import matplotlib.pyplot as plt
import numpy as np
import settings
# import static_atk
# import tiny_taxinet
# import tiny_taxinet2
from loguru import logger
# from tiny_taxinet import process_image
from xplane_screenshot import get_xplane_image
from PIL import Image
import jax.numpy as jnp
import jax
import torch
from torch.utils.data import DataLoader, TensorDataset

import xpc3
import xpc3_helper

from utils.torch2jax import torch2jax
import hj_reachability as hj
from hjnnv import hjnnvUncertaintyAwareFilter
import dynamic_models
import tiny_taxinet2
import json

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *

import matplotlib.pyplot as plt

jax.config.update('jax_platform_name', 'cpu')


def get_perturbed_image(image_raw):
    ptbd_image = settings.PROCESS_IMG(image_raw)

    # Add adversarial attack if applicable
    if settings.ATTACK is not None:
        # image_processed += attack.get_patch(image_processed, 0.032)
        ptbd_image += settings.ATTACK.get_patch(ptbd_image, settings.ATTACK_STRENGTH)

    return ptbd_image


def monte_carlo_bounds(
        image,
        attack_strength,
        network,
        num_samples=int(1e6),
        batch_size=4096,
        device="cuda"
        ):
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    network = settings.NETWORK()

    # Create all perturbations at once
    perturbations = np.random.uniform(
        -attack_strength, attack_strength,
        size=(num_samples, *image.shape)
    )
    samples = image + perturbations

    # Convert to Torch dataset
    dataset = TensorDataset(torch.tensor(samples[:, :, None], dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    network = network.to(device)
    network.eval()

    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(loader):
            batch = batch.to(device)
            outputs = network(batch)  # shape: (B, 2)

            # Move to CPU and update bounds
            outputs_np = outputs.cpu().numpy()
            lo = np.minimum(lo, outputs_np.min(axis=0))
            hi = np.maximum(hi, outputs_np.max(axis=0))

            if batch_idx % max(1, (len(loader) // 10)) == 0:
                print(f"Percent complete: {batch_idx / len(loader) * 100:.2f}%")

    return hj.sets.Box(lo, hi)

    # for i in range(int(num_samples)):
    #     # Sample a random perturbation
    #     perturbation = np.random.uniform(-attack_strength, attack_strength, size=image.shape)
    #     sample = image + perturbation

    #     cte, he = network(torch.Tensor(sample))
    #     # Update bounds
    #     hi = np.maximum(hi, [cte, he])
    #     lo = np.minimum(lo, [cte, he])

    #     if i % (num_samples // 10) == 0:
    #         print(f"Percent complete: {i / num_samples * 100:.2f}%")

    # return hj.sets.Box(lo.flatten(), hi.flatten())


def auto_lirpa_bounds(image, eps):
    bound_opts = {
        # 'relu': "CROWN-IBP",
        'relu': "IBP",
        # 'sparse_intermediate_bounds': False,
        # 'sparse_conv_intermediate_bounds': False,
        # 'sparse_intermediate_bounds_with_ibp': False,
        'sparse_intermediate_bounds': True,
        'sparse_conv_intermediate_bounds': True,
        'sparse_intermediate_bounds_with_ibp': True,
        'sparse_features_alpha': False,
        'sparse_spec_alpha': False,
        'reduce_layers': True,
        # 'zero-lb': True,
        'same-slope': True,
    }
    model = settings.NETWORK()

    if settings.STATE_ESTIMATOR == 'tiny_taxinet':
        x = image.reshape(1, 128, 1)
        dummy_input = torch.empty(1, 128, 1)
    elif settings.STATE_ESTIMATOR == 'dnn':
        x = image.reshape(1, 3, 224, 224)
        dummy_input = torch.empty(1, 3, 224, 224)
    else:
        print("Invalid state estimator name - assuming fully observable")
    
    bounded_model = BoundedModule(
        model,
        dummy_input,
        bound_opts=bound_opts,
        device="cpu"
    )

    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    bounded_image = BoundedTensor(x, ptb)

    lb, ub = bounded_model.compute_bounds(
        x=(bounded_image,),
        method="backward"
    )

    return hj.sets.Box(lb.detach().numpy().flatten(), ub.detach().numpy().flatten())


def main():
    # with xpc3.XPlaneConnect() as client:
    #     sim_speed = 1.0
    #     # Set weather and time of day
    #     client.sendDREF("sim/time/zulu_time_sec", settings.TIME_OF_DAY * 3600 + 8 * 3600)
    #     client.sendDREF("sim/weather/cloud_type[0]", settings.CLOUD_COVER)

    #     client.sendDREF("sim/time/sim_speed", sim_speed)
    #     xpc3_helper.reset(
    #         client,
    #         cteInit=settings.START_CTE,
    #         heInit=settings.START_HE,
    #         dtpInit=settings.START_DTP
    #     )

    # Set up the uncertainty-aware filter
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            np.array([-15., -np.pi/4]),
            np.array([15., np.pi/4])),
        (100, 100),
    )

    if settings.STATE_ESTIMATOR == 'tiny_taxinet':
        hjnnv_filter = hjnnvUncertaintyAwareFilter(
            dynamic_models.TaxiNetDynamics(),
            pred_model=torch2jax(settings.NETWORK()),
            grid=grid,
            num_controls=50,
            num_disturbances=30,
        )

    image_dir = '/home/nick/code/hjnnv/data/verification_test_images/'
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))] if os.path.isdir(image_dir) else []

    if not image_files:
        print("got it")
        image_raw = get_xplane_image()
        os.makedirs(image_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(image_dir, f"captured_{timestamp}.png")
        Image.fromarray(image_raw).save(image_path)
        print(f"Saved image to {image_path}")
    else:
        print("using existing image")
        image_path = os.path.join(image_dir, image_files[0])
        image_raw = np.array(Image.open(image_path))

    ptbd_image = get_perturbed_image(image_raw)
    if settings.STATE_ESTIMATOR == 'tiny_taxinet':
        ptbd_image = ptbd_image.reshape(-1, 1)
    elif settings.STATE_ESTIMATOR == 'dnn':
        ptbd_image = ptbd_image.reshape(-1, 3, 224, 224)
    
    if settings.STATE_ESTIMATOR == 'tiny_taxinet':
        state_bounds = hjnnv_filter.nnv_state_bounds(
            ptbd_image,
            settings.ATTACK_STRENGTH
        )

    # sampled_state_bounds = monte_carlo_bounds(
    #     ptbd_image,
    #     settings.ATTACK_STRENGTH,
    #     settings.NETWORK(),
    #     num_samples=int(1e7),
    #     batch_size=2048
    # )

    auto_lirpa_state_bounds = auto_lirpa_bounds(
        ptbd_image,
        settings.ATTACK_STRENGTH,
    )

    print(state_bounds)
    print(auto_lirpa_state_bounds)

    # print(sampled_state_bounds)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
