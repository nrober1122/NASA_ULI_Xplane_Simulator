import ipdb
from isort import file
import numpy as np
import argparse
import json

import matplotlib.pyplot as plt
import pypalettes
import pickle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


rgb_colors = np.array(pypalettes.load_cmap("Alexandrite").rgb)/255.0
rgb_colors[2] = rgb_colors[2] * 0.8
black = list(rgb_colors[0])
pink = list(rgb_colors[1])
dark_teal = list(rgb_colors[2])
light_teal = list(rgb_colors[3])
purple = list(rgb_colors[4])
orange = list(rgb_colors[5])
yellow = [240/255.0, 225/255.0, 0/255.0]  # yellow
dark_orange = [255/255.0, 103/255.0, 0/255.0]
# gray = [70/255.0, 45/255.0, 255/255.0]  # red
# gray = dark_orange
light_purple = list(rgb_colors[6])
teal = list(rgb_colors[7])
hot_pink = [1.0, 46/255.0, 204/255.0]  # hot pink


data_dir = '/home/nick/code/hjnnv/data/scratch/NASA_ULI_Xplane_Simulator/logs/'


def plot_sim(filename, plot_type, aux_file=None, plot_u=False):
    data_file = data_dir + filename + '/sim2_results.pkl'
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    settings_file = data_dir + filename + '/settings.json'
    with open(settings_file) as f:
        settings = json.load(f)
    
    if plot_type == "states":
        plot_states(data, settings)
    elif plot_type == "trajectory":
        aux_data = None
        if aux_file is not None:
            aux_data_file = data_dir + aux_file + '/sim2_results.pkl'
            with open(aux_data_file, "rb") as f:
                aux_data = pickle.load(f)
        plot_trajectory(data, settings, aux_data, plot_u)
    else:
        raise ValueError("Invalid plot type. Choose 'states' or 'trajectory'.")


    
def plot_states(data, settings):
    T_state_gt = np.stack(data['T_state_gt'], axis=0)[:, [0, 2]]
    T_state_est = np.stack(data['T_state_est'], axis=0)[:, [0, 2]]
    T_state_clean = np.stack(data['T_state_clean'], axis=0)[:, [0, 2]]
    T_rudder = np.stack(data['T_rudder'], axis=0)
    T_rudder_filtered = np.stack(data['T_rudder_filtered'], axis=0)[:, 0]
    T_rudder_unfiltered = np.stack(data['T_rudder_unfiltered'], axis=0)
    T_state_bounds = data.get('T_state_bounds', [])

    lows = np.array([bound.lo for bound in T_state_bounds])
    highs = np.array([bound.hi for bound in T_state_bounds])
    lows_ = lows + np.array([settings["CTE_BUFFER"], np.deg2rad(settings["HE_BUFFER"])])
    highs_ = highs - np.array([settings["CTE_BUFFER"], np.deg2rad(settings["HE_BUFFER"])])


    T_t = np.arange(T_state_gt.shape[0]) * settings["DT"]
    labels = ["CTE (m)", "HE (degrees)"]

    linewidth = 1.5

    fig, axes = plt.subplots(2, layout="constrained", figsize=(12, 8), sharex=True)
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_state_gt[:, ii], color=dark_teal, label="True", linewidth=linewidth)
        ax.plot(T_t, T_state_est[:, ii], color=pink, label="Estimated", linewidth=linewidth)
        ax.plot(T_t, T_state_clean[:, ii], color=purple, label="No attack", linewidth=linewidth)
        if ii == 0:
            ax.plot(T_t, -10*np.ones_like(T_t), color=black, label="Safe Boundary", linewidth=linewidth, linestyle='--')
            ax.plot(T_t, 10*np.ones_like(T_t), color=black, linewidth=linewidth, linestyle='--')
        if settings["FILTER"]:
            if ii == 1:
                lows[:, ii] = np.rad2deg(lows[:, ii])
                highs[:, ii] = np.rad2deg(highs[:, ii])
                lows_[:, ii] = np.rad2deg(lows_[:, ii])
                highs_[:, ii] = np.rad2deg(highs_[:, ii])
            ax.fill_between(T_t, lows[:, ii], highs[:, ii], color=light_teal, alpha=0.4, label="Inflated Bounds" if ii == 0 else None)
            ax.fill_between(T_t, lows_[:, ii], highs_[:, ii], color=teal, alpha=0.4, label="NNV Bounds" if ii == 0 else None)
        

        # ax.plot(T_t, T_state_gt[:, ii]+2, color=rgb_colors[1], label="True", linewidth=4)
        # ax.plot(T_t, T_state_gt[:, ii]+4, color=rgb_colors[2], label="True", linewidth=4)
        # ax.plot(T_t, T_state_gt[:, ii]+6, color=rgb_colors[3], label="True", linewidth=4)
        # ax.plot(T_t, T_state_gt[:, ii]+8, color=rgb_colors[4], label="True", linewidth=4)
        # ax.plot(T_t, T_state_gt[:, ii]+10, color=rgb_colors[5], label="True", linewidth=4)
        # ax.plot(T_t, T_state_gt[:, ii]+12, color=rgb_colors[6], label="True", linewidth=4)
        # ax.plot(T_t, T_state_gt[:, ii]+14, color=rgb_colors[7], label="True", linewidth=4)        
        # ax.plot(T_t, T_state_gt[:, ii]+2, color=rgb_colors[3], label="True", linewidth=4)
        # ax.plot(T_t, T_state_est[:, ii]+2, color=rgb_colors[4], label="Estimated", linewidth=4)
        # ax.plot(T_t, T_state_clean[:, ii]+2, color=rgb_colors[5], label="No attack", linewidth=4)
        # ax.plot(T_t, T_state_gt[:, ii]+4, color=rgb_colors[6], label="True", linewidth=4)
        # ax.plot(T_t, T_state_est[:, ii]+4, color=rgb_colors[7], label="Estimated", linewidth=4)
        # ax.plot(T_t, T_state_clean[:, ii]+4, color=rgb_colors[8], label="No attack", linewidth=4)
        ax.set_ylabel(labels[ii], rotation=0, ha="right")
    leg = axes[0].legend(framealpha=1.0, edgecolor="0.3")
    # leg.get_frame().set_facecolor("white")
    axes[1].set_xlabel("Time (s)")
    # fig.savefig(results_dir + "sim2_traj.pdf")
    plt.show()
    plt.close(fig)
    # import ipdb; ipdb.set_trace()

def plot_trajectory(data, settings, aux_data=None, plot_u=False):
    T_state_gt = np.stack(data['T_state_gt'], axis=0)
    T_state_est = np.stack(data['T_state_est'], axis=0)
    T_state_clean = np.stack(data['T_state_clean'], axis=0)
    T_rudder = np.stack(data['T_rudder'], axis=0)
    T_rudder_filtered = np.stack(data['T_rudder_filtered'], axis=0)[:, 0]
    T_rudder_unfiltered = np.stack(data['T_rudder_unfiltered'], axis=0)
    T_state_bounds = data.get('T_state_bounds', [])

    T_state_unfiltered = np.stack(aux_data['T_state_gt'], axis=0) if aux_data is not None else None

    lows = np.array([bound.lo for bound in T_state_bounds])
    highs = np.array([bound.hi for bound in T_state_bounds])

    T_t = np.arange(T_state_gt.shape[0]) * settings["DT"]
    labels = ["CTE (m)", "HE (degrees)"]
    print("Plotting trajectory")
    cte_constr = 10.0
    ylim = 11.0

    def plot_trajectory_helper(ax, T_state, lows, highs, T_state_unfiltered=None, cax=None):
        hot_color = hot_pink
        cold_color = black
        bounds_color = teal
        bounds_alpha = 0.6
        boundary_color = dark_teal

        print([c*255.0 for c in rgb_colors])
        

        # Compute the color values based on abs(rudder_unfiltered - rudder_filtered)
        color_vals = np.abs(T_rudder_unfiltered - T_rudder_filtered)
        color_vals = np.convolve(color_vals, np.ones(5)/5, mode='same')

        norm = plt.Normalize(vmin=np.min(color_vals), vmax=np.max(color_vals))
        
        # Create custom colormap
        colors = [cold_color, hot_color]
        cmap = mcolors.LinearSegmentedColormap.from_list("dark_teal_2_purple", colors)

        # Create a colored line using LineCollection

        # points = np.array([T_state_gt[:, 1], T_state_gt[:, 0]]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # # lc = LineCollection(segments, cmap=cmap, norm=norm, array=color_vals, linewidth=3, zorder=11, label="Position")
        # # ax.add_collection(lc)
        # sc = ax.scatter(T_state_gt[:, 1], T_state_gt[:, 0], c=color_vals, cmap=cmap, norm=norm, s=10, zorder=11)
        # plt.colorbar(sc, ax=ax, label="|rudder_unfiltered - rudder_filtered|")

        # Create colored line segments
        points = np.array([T_state_gt[:, 1], T_state_gt[:, 0]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            array=color_vals[:-1],  # one color per segment
            linewidth=4.0,
            zorder=11,
        )
        lc.set_capstyle('round')
        lc.set_joinstyle('round')
        line = ax.add_collection(lc)
        if T_state_unfiltered is not None:
            ax.plot(T_state_unfiltered[:, 1],
                    T_state_unfiltered[:, 0],
                    color=purple,
                    linestyle='--',
                    label="Unfiltered",
                    linewidth=2,
                    zorder=10
                    )

        # Add colorbar
        # --- Create aligned, slightly shorter colorbar ---
        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.1)

            # Create the colorbar manually
            cbar = plt.colorbar(line, cax=cax)

            # Tweak appearance
            cbar.ax.tick_params(labelsize=10, width=1)  # tick font & thickness
            cbar.set_label("|rudder_unfiltered - rudder_filtered|", fontsize=11, labelpad=6)

            # Make the colorbar slightly shorter than the main axis
            # (this uses transform math to crop the cbar height)
            pos = cax.get_position()
            shorten = 0.06  # fraction of height to trim from top and bottom
            cax.set_position([pos.x0, pos.y0 + shorten/2 * pos.height, pos.width, pos.height * (1 - shorten)])

            # Optional: add a subtle rounded border
            for spine in cbar.ax.spines.values():
                spine.set_linewidth(1)
                spine.set_color("0.3")
        else:
            # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            # cbar = plt.colorbar(sm, cax=cax)
            # cbar.set_label("|rudder_unfiltered - rudder_filtered|", fontsize=11)
            # Create the colorbar manually
            cbar = plt.colorbar(line, cax=cax)

            pos = cax_top.get_position()
            shorten = 0.32  # 5% height reduction
            cax_top.set_position([
                pos.x0,
                pos.y0 + shorten * pos.height / 2,
                pos.width,
                pos.height * (1 - shorten),
            ])

            # Tweak appearance
            cbar.ax.tick_params(labelsize=10, width=1)  # tick font & thickness
            cbar.set_label("|rudder_unfiltered - rudder_filtered|", fontsize=11, labelpad=6)

            # Make the colorbar slightly shorter than the main axis
            # (this uses transform math to crop the cbar height)
            pos = cax.get_position()
            shorten = 0.06  # fraction of height to trim from top and bottom
            cax.set_position([pos.x0, pos.y0 + shorten/2 * pos.height, pos.width, pos.height * (1 - shorten)])

            # Optional: add a subtle rounded border
            for spine in cbar.ax.spines.values():
                spine.set_linewidth(1)
                spine.set_color("0.3")

        ax.fill_between(T_state_gt[:, 1], lows[:, 0], highs[:, 0], color=bounds_color, alpha=bounds_alpha, label="CTE Bounds", zorder=9)
        ax.set_aspect("equal")
        # ymin, ymax = ax.get_ylim()
        # ymin, ymax = min(ymin, -ylim), max(ymax, ylim)
        ax.set_ylim(-ylim, ylim)
        ax.axhspan(cte_constr, ylim, color=boundary_color, alpha=1.0)
        ax.axhspan(-cte_constr, -ylim, color=boundary_color, alpha=1.0)
    
    if not plot_u:
        fig, ax = plt.subplots(figsize=(18, 6))
        plot_trajectory_helper(ax, T_state_gt, lows, highs, T_state_unfiltered)
    else:

        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(
            2,  # two rows (top and bottom)
            2,  # two columns (main plot + colorbar)
            width_ratios=[40, 1],  # make colorbar narrower
            height_ratios=[2, 1],  # equal height
            wspace=0.05,  # small horizontal gap
            hspace=0.1,   # small vertical gap
        )

        # --- Top plot (with colorbar) ---
        ax_top = fig.add_subplot(gs[0, 0])
        cax_top = fig.add_subplot(gs[0, 1])  # colorbar axis

        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)

        plot_trajectory_helper(ax_top, T_state_gt, lows, highs, T_state_unfiltered, cax=cax_top)

        ax_bottom.plot(T_state_gt[:, 1], T_rudder_filtered, color=pink, label="Rudder (filtered)", linewidth=1.5)
        ax_bottom.plot(T_state_gt[:, 1], T_rudder_unfiltered, color=dark_teal, label="Rudder (unfiltered)", linewidth=1.5)
        ax_bottom.set_ylabel("Rudder (degrees)", rotation=0, ha="right")
        ax_bottom.set_xlabel("DTP (m)")
        ax_bottom.legend(loc="lower left")
        ax_bottom.grid(True)
    plt.show()
    # plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results.")
    parser.add_argument("--file", help="Path to the data file to load")
    parser.add_argument("--aux_file", help="Path to the auxiliary data file to load", default=None)
    parser.add_argument("--plot_type", help="Type of plot to generate", choices=["states", "trajectory"])
    parser.add_argument("--plot_u", action="store_true", default=False, help="Whether to plot control inputs")
    args = parser.parse_args()

    plot_sim(args.file, args.plot_type, aux_file=args.aux_file, plot_u=args.plot_u)
