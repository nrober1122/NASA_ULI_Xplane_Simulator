import os
import pathlib
import pickle

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from simulation.sim_result import SimResult

NASA_ULI_ROOT_DIR = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"])
DATA_DIR = pathlib.Path(os.environ["NASA_DATA_DIR"])
SCRATCH_DIR = NASA_ULI_ROOT_DIR / "scratch"
PLOT_DIR = SCRATCH_DIR / "stride_results/plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    S_strides = np.array([1, 2, 4, 8, 16])
    S_results: list[list[SimResult]] = []

    linfnorms = np.linspace(0.0, 0.035, num=10)

    for stride in [1, 2, 4, 8, 16]:
        pkl_path = SCRATCH_DIR / f"stride_results/data_{stride}.pkl"
        with open(pkl_path, "rb") as f:
            results_stride: list[SimResult] = pickle.load(f)
            S_results.append(results_stride)

    # 1: Plot maximum CTE at the final timestep as a function of the linfnorm.
    max_cte_dict = {infnorm: [] for infnorm in linfnorms}
    for ss, stride in enumerate([1, 2, 4, 8, 16]):
        norm_results: list[SimResult] = S_results[ss]
        assert len(norm_results) == len(linfnorms)
        for jj, result in enumerate(norm_results):

            if ss == 0:
                print(result.T_state_gt[:, 0])
                ipdb.set_trace()

            max_cte = np.abs(result.T_state_gt[-10:, 0]).max()
            max_cte_dict[linfnorms[jj]].append(max_cte)

    cmap = sns.color_palette("flare", as_cmap=True)

    figsize = np.array([5.0, 3.0])
    fig, ax = plt.subplots(layout="constrained", figsize=figsize, dpi=500)

    for jj, infnorm in enumerate(linfnorms[::-1]):
        alpha = 1.0 - jj / (len(linfnorms) - 1)
        color = cmap(alpha)
        S_max_cte = max_cte_dict[infnorm]
        label = r"$\Vert \cdot \Vert_\infty = {:.3f}$".format(infnorm)
        ax.plot(S_strides, S_max_cte, color=color, alpha=0.8, label=label)

    ax.set_xscale("log", base=2)
    formatter = FuncFormatter(lambda y, _: "{:d}".format(int(y)))
    ax.xaxis.set_major_formatter(formatter)
    ax.set(xlabel="Downscale Factor", ylabel="Maximum CTE (m)")
    # ax.legend()

    cm = ScalarMappable(norm=Normalize(linfnorms[0], linfnorms[-1]), cmap=cmap)
    cbar = fig.colorbar(cm, ax=ax, orientation="vertical")
    cbar.ax.set_ylabel(r"$\Vert \cdot \Vert_\infty$", rotation=0, ha="left")

    fig_path = PLOT_DIR / "max_cte_stride.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight")
    ################################################################


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
