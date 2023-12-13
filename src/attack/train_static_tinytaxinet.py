import os
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.colors import CenteredNorm
from torch.utils.data import DataLoader

from simulation.controllers import get_p_control_torch, getProportionalControl
from simulation.tiny_taxinet2 import get_network
from tiny_taxinet_train.model_tiny_taxinet import TinyTaxiNetDNN
from tiny_taxinet_train.tiny_taxinet_dataloader import tiny_taxinet_prepare_dataloader

NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]
DATA_DIR = os.environ["NASA_ULI_DATA_DIR"]

# OBJECTIVE = "mse"
OBJECTIVE = "lyap"

class TinyTaxiNetAttackStatic(torch.nn.Module):
    def __init__(self, image_size: tuple[int, int], max_delta: float, rudder_target: float):
        super().__init__()
        self.patch = torch.nn.Parameter(torch.zeros(image_size, dtype=torch.float32))
        self.register_buffer("max_delta", torch.tensor(max_delta))
        self.register_buffer("rudder_target", torch.tensor(rudder_target))
        self.max_delta = self.max_delta
        self.rudder_target = self.rudder_target

    def clip_patch(self):
        self.patch.data = torch.clip(self.patch.data, -self.max_delta, self.max_delta)
        return self


def train_model_atk(
    model: TinyTaxiNetDNN,
    model_atk: TinyTaxiNetAttackStatic,
    dl_train: DataLoader,
    dl_test: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    results_dir,
    num_epochs=25,
    log_every=100,
):
    for epoch in range(num_epochs):
        # Train.
        model_atk.train()
        loss_mses_train, loss_mses_eval = [], []
        for inputs, _ in dl_train:
            inputs = inputs.to(device)
            batch_size = inputs.shape[0]

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # (b, 2)
                b_outputs = model(inputs + model_atk.patch)
                b_cte, b_he = b_outputs[:, 0], b_outputs[:, 1]
                b_rudder = get_p_control_torch(b_cte, b_he)
                assert b_rudder.shape == (batch_size,)

                if OBJECTIVE == "mse":
                    # We want the rudder to be equal to the desired value.
                    loss_mse = torch.mean((b_rudder - model_atk.rudder_target) ** 2)
                elif OBJECTIVE == "lyap":
                    # We want to maximize (theta * omega) ~ (-theta * rudder).
                    Vdot_approx = -torch.mean(b_he * b_rudder)
                    loss_mse = -Vdot_approx
                else:
                    raise NotImplementedError("")

                loss_mse.backward()
                optimizer.step()
            loss_mses_train.append(loss_mse.detach().item())

            # Project the gradient to within reasonable values.
            model_atk = model_atk.clip_patch()

        # Eval.
        with torch.inference_mode():
            model_atk.eval()
            for inputs, _ in dl_test:
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]

                b_outputs = model(inputs + model_atk.patch)
                b_cte, b_he = b_outputs[:, 0], b_outputs[:, 1]
                b_rudder = get_p_control_torch(b_cte, b_he)
                assert b_rudder.shape == (batch_size,)

                if OBJECTIVE == "mse":
                    # We want the rudder to be equal to the desired value.
                    loss_mse = torch.mean((b_rudder - model_atk.rudder_target) ** 2)
                elif OBJECTIVE == "lyap":
                    # We want to maximize (theta * omega) ~ (-theta * rudder).
                    omega_target = 0.2 * torch.sign(b_he)
                    rudder_target = -omega_target
                    loss_mse = torch.mean((b_rudder - rudder_target) ** 2)
                else:
                    raise NotImplementedError("")

                loss_mses_eval.append(loss_mse.item())

        # Log.
        loss_mses_train = np.array(loss_mses_train)
        loss_mses_eval = np.array(loss_mses_eval)
        logger.info("Train: {:6.2e}, Test: {:6.2e}".format(loss_mses_train.mean(), loss_mses_eval.mean()))

    return model_atk


def main():
    scratch_dir = pathlib.Path(NASA_ULI_ROOT_DIR) / "scratch"
    models_dir = pathlib.Path(NASA_ULI_ROOT_DIR) / "models"
    results_dir = scratch_dir / f"tiny_taxinet_attack_static_{OBJECTIVE}"
    results_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dl_params = {"batch_size": 256, "shuffle": True, "num_workers": 1, "drop_last": False, "pin_memory": True}

    model = get_network()
    model.eval()
    model.to(device)

    # Freeze the network.
    for param in model.parameters():
        param.requires_grad = False

    # Create the attack model.
    image_size = (8, 16)
    max_delta = 0.027
    rudder_target = 1.0
    model_atk = TinyTaxiNetAttackStatic(image_size, max_delta, rudder_target)
    model_atk.to(device)

    # Create the dataloaders.
    cond_list = ["morning"]
    dset_train, dl_train = tiny_taxinet_prepare_dataloader(DATA_DIR, cond_list, "train", dl_params)
    dset_test, dl_test = tiny_taxinet_prepare_dataloader(DATA_DIR, cond_list, "validation", dl_params)
    optimizer = torch.optim.Adam(model_atk.parameters(), lr=5e-4, fused=True)

    model_atk = train_model_atk(model, model_atk, dl_train, dl_test, device, optimizer, results_dir)

    # Save the model.
    torch.save(model_atk.state_dict(), results_dir / "model_atk.pt")

    # Visualize the patch, the image and the patch + image.
    # Also show the resulting state estimate and controls below the image.
    image_raw = dset_test[0][0].cpu().numpy()
    patch_np = model_atk.patch.detach().cpu().numpy()
    assert patch_np.shape == image_raw.shape
    image_atked = (image_raw + patch_np).clip(0, 1)

    cte_raw, he_raw = model(torch.tensor(image_raw[None, :, None], device=device)).detach().cpu().numpy().squeeze()
    rudder_raw = getProportionalControl(None, cte_raw, he_raw)

    cte_atk, he_atk = model(torch.tensor(image_atked[None, :, None], device=device)).detach().cpu().numpy().squeeze()
    rudder_atk = getProportionalControl(None, cte_atk, he_atk)

    raw_descr = "CTE: {:.2f}, HE: {:.2f}\nRudder: {:.2f}".format(cte_raw, he_raw, rudder_raw)
    atk_descr = "CTE: {:.2f}, HE: {:.2f}\nRudder: {:.2f}".format(cte_atk, he_atk, rudder_atk)

    figsize = 2 * np.array([3 * 2, 1])
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    im = axes[0].imshow(patch_np, norm=CenteredNorm(), cmap="RdBu_r")
    fig.colorbar(im, ax=axes[0])
    axes[1].imshow(image_raw, cmap="gray")
    axes[2].imshow(image_atked, cmap="gray")
    axes[0].set_title("Learned Patch")
    axes[1].set_title("Raw Image\n{}".format(raw_descr))
    axes[2].set_title("Perturbed Image\n{}".format(atk_descr))
    [ax.axis("off") for ax in axes]
    [ax.grid(False) for ax in axes]
    fig.savefig(results_dir / "patch.pdf", bbox_inches="tight")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
