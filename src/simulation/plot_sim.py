import numpy as np
import argparse
import json

import matplotlib.pyplot as plt
import pypalettes


rgb_colors = np.array(pypalettes.load_cmap("Alexandrite").rgb)/255.0
black = rgb_colors[0]
pink = rgb_colors[1]
dark_teal = rgb_colors[2]
light_teal = rgb_colors[3]
purple = rgb_colors[4]
orange = rgb_colors[5]
light_purple = rgb_colors[6]
teal = rgb_colors[7]


data_dir = '/home/nick/code/hjnnv/data/scratch/NASA_ULI_Xplane_Simulator/logs/'


def plot_sim(data_dir, plot_type):
    data_file = data_dir + '/sim2_data.npz'
    data = np.load(data_file)

    settings_file = data_dir + '/settings.json'
    with open(settings_file) as f:
        settings = json.load(f)

    
def plot_states(data, settings):
    T_state_gt = data['T_state_gt'][:-1][:, [0, 2]]
    T_state_est = data['T_state_est'][:-1][:, [0, 2]]
    T_state_clean = data['T_state_clean'][:-1][:, [0, 2]]
    T_rudder = data['T_rudder'][:-1]
    T_rudder_filtered = data['T_rudder_filtered'][:-1]

    T_t = np.arange(T_state_gt.shape[0]) * settings["DT"]
    labels = ["CTE (m)", "HE (degrees)"]

    fig, axes = plt.subplots(2, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_state_gt[:, ii], color=rgb_colors[0], label="True", linewidth=4)
        ax.plot(T_t, T_state_gt[:, ii]+2, color=rgb_colors[1], label="True", linewidth=4)
        ax.plot(T_t, T_state_gt[:, ii]+4, color=rgb_colors[2], label="True", linewidth=4)
        ax.plot(T_t, T_state_gt[:, ii]+6, color=rgb_colors[3], label="True", linewidth=4)
        ax.plot(T_t, T_state_gt[:, ii]+8, color=rgb_colors[4], label="True", linewidth=4)
        ax.plot(T_t, T_state_gt[:, ii]+10, color=rgb_colors[5], label="True", linewidth=4)
        ax.plot(T_t, T_state_gt[:, ii]+12, color=rgb_colors[6], label="True", linewidth=4)
        ax.plot(T_t, T_state_gt[:, ii]+14, color=rgb_colors[7], label="True", linewidth=4)

        
        # ax.plot(T_t, T_state_est[:, ii], color=rgb_colors[1], label="Estimated", linewidth=4)
        # ax.plot(T_t, T_state_clean[:, ii], color=rgb_colors[2], label="No attack", linewidth=4)
        # ax.plot(T_t, T_state_gt[:, ii]+2, color=rgb_colors[3], label="True", linewidth=4)
        # ax.plot(T_t, T_state_est[:, ii]+2, color=rgb_colors[4], label="Estimated", linewidth=4)
        # ax.plot(T_t, T_state_clean[:, ii]+2, color=rgb_colors[5], label="No attack", linewidth=4)
        # ax.plot(T_t, T_state_gt[:, ii]+4, color=rgb_colors[6], label="True", linewidth=4)
        # ax.plot(T_t, T_state_est[:, ii]+4, color=rgb_colors[7], label="Estimated", linewidth=4)
        # ax.plot(T_t, T_state_clean[:, ii]+4, color=rgb_colors[8], label="No attack", linewidth=4)
        ax.set_ylabel(labels[ii], rotation=0, ha="right")
    axes[0].legend()
    # fig.savefig(results_dir + "sim2_traj.pdf")
    plt.show()
    plt.close(fig)
    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results.")
    parser.add_argument("file", help="Path to the data file to load")
    parser.add_argument("plot_type", help="Type of plot to generate", choices=["states, trajectory"])
    args = parser.parse_args()

    data_dir = data_dir + args.file
    plot_sim(data_dir, args.plot_type)
