import os
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from simulation.static_atk import StaticAttack

NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])
SCRATCH_DIR = NASA_ULI_ROOT_DIR / "scratch"
PLOT_DIR = SCRATCH_DIR / "stride_results/plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    S_strides = [1, 2, 4, 8, 16]
    S = len(S_strides)

    patches = []
    for stride in S_strides:
        attack = StaticAttack(stride=stride)
        nn = attack.get_network()
        patch = nn.patch.detach().cpu().numpy()
        patches.append(patch)

    delta = 0.03
    norm = Normalize(vmin=-delta, vmax=delta)

    figsize = np.array([2 * S, 2])
    fig, axes = plt.subplots(1, S, figsize=figsize, layout="constrained", dpi=300)
    for ii, ax in enumerate(axes):
        im = ax.imshow(patches[ii], cmap="RdBu_r", norm=norm)
    fig.colorbar(im, ax=ax)
    [ax.axis("off") for ax in axes]
    [ax.grid(False) for ax in axes]
    fig_path = PLOT_DIR / "learned_patches.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
