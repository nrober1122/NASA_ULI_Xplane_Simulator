import ipdb
from isort import file
import numpy as np
import jax.numpy as jnp
import argparse
import json

import matplotlib.pyplot as plt
import pypalettes
import pickle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import hj_reachability as hj
import dynamic_models

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

plt.rcParams['font.family'] = 'serif'          # e.g., 'sans-serif', 'serif', 'monospace'
plt.rcParams['font.serif'] = ['Times New Roman']  # specify font
plt.rcParams['font.size'] = 32                 # set global font size
mpl.rcParams["text.usetex"] = True
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'  # optional, for math text consistency

data_dir = '/home/nick/code/hjnnv/data/scratch/NASA_ULI_Xplane_Simulator/logs/'


def plot_sim(args):
    data_file = data_dir + args.file + '/sim2_results.pkl'
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    settings_file = data_dir + args.file + '/settings.json'
    with open(settings_file) as f:
        settings = json.load(f)

    if args.plot_type == "states":
        plot_states(data, settings)
    elif args.plot_type == "trajectory":
        aux_data = None
        if args.aux_file is not None:
            aux_data_file = data_dir + args.aux_file + '/sim2_results.pkl'
            with open(aux_data_file, "rb") as f:
                aux_data = pickle.load(f)
        plot_trajectory(data, settings, aux_data, args.plot_u, args.cbar)
    elif args.plot_type == "value_function":
        plot_value_function(data, settings)
    elif args.plot_type == "images":
        plot_images(data, settings)
    else:
        raise ValueError("Invalid plot type. Choose 'states', 'trajectory', 'value_function', or 'images'.")


def plot_states(data, settings):
    plt.rcParams['font.size'] = 20
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

    boundary_color = orange
    boundary_alpha = 0.4
    nnv_bounds_color = teal
    inflated_bounds_color = light_teal
    nnv_bounds_alpha = 0.3
    inflated_bounds_alpha = 0.3


    linewidth = 1.5

    fig, axes = plt.subplots(2, layout="constrained", figsize=(12, 6), sharex=True)
    for ii, ax in enumerate(axes):
        true_label = r"$\mathbf{x}_{t}$"
        estimated_label = r"$\tilde{\hat{\mathbf{x}}}_{t}$"
        clean_label = r"$\hat{\mathbf{x}}_{t}$"
        ax.plot(T_t, T_state_gt[:, ii], color=dark_teal, label=true_label, linewidth=linewidth)
        ax.plot(T_t, T_state_est[:, ii], color=pink, label=estimated_label, linewidth=linewidth)
        ax.plot(T_t, T_state_clean[:, ii], color=purple, label=clean_label, linewidth=linewidth)
        legend_ax = 0
        if ii == 0:
            cte_constr = 10.0
            ylim = 12.0
            lower_rect = plt.Rectangle(
                (-2, -ylim-2),
                T_t[-1]+3, ylim-cte_constr+2,
                linewidth=1.5,
                edgecolor=black,
                facecolor=boundary_color + [boundary_alpha],
                linestyle='-.',
                # label='Unsafe Region',
                zorder=0)
            ax.add_patch(lower_rect)
            upper_rect = plt.Rectangle(
                (-2, cte_constr),
                T_t[-1]+3, ylim-cte_constr+2,
                linewidth=1.5,
                edgecolor=black,
                facecolor=boundary_color + [boundary_alpha],
                linestyle='-.',
                zorder=0)
            ax.add_patch(upper_rect)
            # ax.plot(T_t, -cte_constr*np.ones_like(T_t), color=black, linewidth=linewidth, linestyle='-.')
            # ax.plot(T_t, cte_constr*np.ones_like(T_t), color=black, linewidth=linewidth, linestyle='-.')
            # ax.axhspan(cte_constr, ylim, color=boundary_color, alpha=boundary_alpha,)
            # ax.axhspan(-cte_constr, -ylim, color=boundary_color, alpha=boundary_alpha)
            ax.set_ylim(-ylim, ylim)
        if settings["FILTER"]:
            if ii == 1:
                lows[:, ii] = np.rad2deg(lows[:, ii])
                highs[:, ii] = np.rad2deg(highs[:, ii])
                lows_[:, ii] = np.rad2deg(lows_[:, ii])
                highs_[:, ii] = np.rad2deg(highs_[:, ii])
                import ipdb; ipdb.set_trace()
            ax.fill_between(T_t, lows_[:, ii], highs_[:, ii], color=nnv_bounds_color, alpha=nnv_bounds_alpha, label="NNV Bounds" if ii == legend_ax else None)
            ax.fill_between(T_t, lows[:, ii], highs[:, ii], color=inflated_bounds_color, alpha=inflated_bounds_alpha, label=r"$\mathbf{e}_{\hat{\mathbf{x}}}$" if ii == legend_ax else None, zorder=1)
            
        # ax.set_ylabel(labels[ii], rotation=0, ha="right")
        ax.set_ylabel(labels[ii])
        ax.set_xlim(0, T_t[-1])
    leg = axes[legend_ax].legend(
        framealpha=1.0,
        edgecolor="0.3",
        fontsize=20,
        loc="lower right",
        ncol=5,
        handlelength=1.0,
        columnspacing=1.0
        )
    # leg = axes[1].legend(framealpha=1.0, edgecolor="0.3", fontsize=16, loc="upper right")
    # leg.get_frame().set_facecolor("white")
    axes[1].set_xlabel("Time (s)")
    # fig.savefig(results_dir + "sim2_traj.pdf")
    plt.show()
    plt.close(fig)
    # import ipdb; ipdb.set_trace()

def plot_trajectory(data, settings, aux_data=None, plot_u=False, plot_cbar=False):
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

    filter_dyn = dynamic_models.TaxiNetDynamics(
        dt=settings["DT"],
        max_rudder=jnp.tan(jnp.deg2rad(settings["MAX_RUDDER"]))
    )
    x_mins = filter_dyn.step(jnp.array(lows), jnp.array([T_rudder_unfiltered]).T, jnp.zeros_like(lows))
    x_maxs = filter_dyn.step(jnp.array(highs), jnp.array([T_rudder_unfiltered]).T, jnp.zeros_like(highs))
    # import ipdb; ipdb.set_trace()

    T_t = np.arange(T_state_gt.shape[0]) * settings["DT"]
    labels = ["CTE (m)", "HE (degrees)"]
    print("Plotting trajectory")
    cte_constr = 10.0
    ylim = 12.0

    def plot_trajectory_helper(ax, T_state, lows, highs, other_lows=None, other_highs=None, T_state_unfiltered=None, cax=None, plot_cbar=False):
        hot_color = hot_pink
        cold_color = black
        filtered_color = 'k'
        filtered_linewidth = 2.0
        unfiltered_color = purple
        unfiltered_linewidth = 2.0
        bounds_color = teal
        bounds_alpha = 0.2
        bounds_line_color = teal
        bounds_line_alpha = 0.8
        boundary_color = orange
        boundary_alpha = 0.4
        boundary_line_color = black
        boundary_line_width = 1.5

        print([c*255.0 for c in rgb_colors])
        
        if plot_cbar:
            # Compute the color values based on abs(rudder_unfiltered - rudder_filtered)
            color_vals = np.abs(T_rudder_unfiltered - T_rudder_filtered)
            color_vals = np.convolve(color_vals, np.ones(3)/3, mode='same')

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
                        color=unfiltered_color,
                        linestyle='--',
                        label="Unfiltered",
                        linewidth=2,
                        zorder=10
                        )
                ax.scatter(T_state_unfiltered[-2, 1], T_state_unfiltered[-2, 0], color=unfiltered_color, marker='X', s=200, zorder=11)

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
        else:
            num_samples = int(1e5)
            sample_states = np.zeros((T_state_gt.shape[0], num_samples, T_state_gt[:, [0, 2]].shape[1]))
            # sample_states = np.zeros((T_state_gt.shape[0], num_samples, T_state_gt.shape[1]))
            filter_dyn = dynamic_models.TaxiNetDynamics(
                dt=settings["DT"],
                max_rudder=jnp.tan(jnp.deg2rad(settings["MAX_RUDDER"]))
            )
            sample_states[0, :, :] = np.random.uniform(low=lows[0, :], high=highs[0, :], size=(num_samples, T_state_gt[:, [0, 2]].shape[1]))
            # sample_states[0, :, 1] = T_state_gt[0, 1]
            # sample_states[0, :, 1] = jnp.deg2rad(T_state_gt[0, 2])  # initialize all samples to the true initial DTP
            for i in range(1, T_state_gt.shape[0]):
                disturbances = np.random.uniform(
                    low=filter_dyn.disturbance_space.lo,
                    high=filter_dyn.disturbance_space.hi,
                    size=(num_samples, 2)
                )
                sample_states[i, :, :] = filter_dyn.step(
                    jnp.array(sample_states[i-1, :, :]),
                    jnp.tan(jnp.deg2rad(T_rudder_filtered[i-1]))*jnp.ones((num_samples, 1)),
                    jnp.array(disturbances)
                )
                sample_states[i, :, :] = jnp.clip(sample_states[i, :, :], jnp.array(lows[i, :]), jnp.array(highs[i, :]))
            
            # import pdb; pdb.set_trace()
            
            ax.plot(T_state_gt[:, 1],
                    T_state_gt[:, 0],
                    color=filtered_color,
                    label=r"\textbf{GUARDIAN} (\textbf{true state} $x_t$)",
                    linewidth=filtered_linewidth,
                    zorder=10
                    )
            # import ipdb; ipdb.set_trace()
            if T_state_unfiltered is not None:
                ax.plot(T_state_unfiltered[:, 1],
                        T_state_unfiltered[:, 0],
                        color=unfiltered_color,
                        linestyle='--',
                        label=r"\textbf{No Filter} (\textbf{true state} $x_t$)",
                        linewidth=unfiltered_linewidth,
                        zorder=10
                        )
                ax.scatter(T_state_unfiltered[-2, 1], T_state_unfiltered[-2, 0], color=unfiltered_color, marker='X', s=200, zorder=11)
            
            ax.plot(T_state_gt[[0,-1], 1],
                    [20, 20],
                    color=light_teal,
                    label=r"\textbf{Sample trajectories of} $x_t$",
                    linewidth=filtered_linewidth,
                    zorder=8,
                    alpha=1.0
                    )
            ax.plot(T_state_gt[:, 1],
                    sample_states[:, :, 0],
                    color=light_teal,
                    # label="State Samples",
                    linewidth=filtered_linewidth*0.1,
                    zorder=8,
                    alpha=0.2
                    )

        ax.fill_between(T_state_gt[:, 1], lows[:, 0], highs[:, 0], color=bounds_color, alpha=bounds_alpha, zorder=9, label=r"\textbf{State bounds} $\bar{\mathcal{X}}_{t}$")
        if other_lows is not None and other_highs is not None:
            ax.fill_between(T_state_gt[:, 1], other_lows[:, 0], other_highs[:, 0], color=light_purple, alpha=bounds_alpha, label="State Bounds", zorder=8)
        ax.plot(T_state_gt[:, 1], highs[:, 0], color=bounds_line_color, alpha=bounds_line_alpha, linewidth=0.7, zorder=9)
        ax.plot(T_state_gt[:, 1], lows[:, 0], color=bounds_line_color, alpha=bounds_line_alpha, linewidth=0.7, zorder=9)
        
        # ax.plot(T_state_gt[:, 1], np.zeros_like(T_state_gt[:, 1]), color=light_purple, linestyle='--', linewidth=0.5, label="Centerline", zorder=8)
        ax.set_aspect("equal")
        # ymin, ymax = ax.get_ylim()
        # ymin, ymax = min(ymin, -ylim), max(ymax, ylim)
        ax.set_xlabel("DTP (m)")
        ax.set_ylabel("CTE (m)")
        ax.set_ylim(-ylim, ylim)
        ax.set_xlim(T_state_gt[0, 1], T_state_gt[-1, 1])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.axhspan(cte_constr, ylim, color=boundary_color, alpha=boundary_alpha)
        ax.axhspan(-cte_constr, -ylim, color=boundary_color, alpha=boundary_alpha)
        ax.plot(T_state_gt[:, 1], cte_constr * np.ones_like(T_state_gt[:, 1]), color=boundary_line_color, linestyle='-.', linewidth=boundary_line_width)
        ax.plot(T_state_gt[:, 1], -cte_constr * np.ones_like(T_state_gt[:, 1]), color=boundary_line_color, linestyle='-.', linewidth=boundary_line_width)
        # Collect legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        
        # Replace the sample trajectories line with a rectangular patch
        sample_traj_idx = labels.index(r"\textbf{Sample trajectories of} $x_t$")
        handles[sample_traj_idx] = mpl.patches.Patch(
            facecolor=light_teal,
            alpha=1.0,
            edgecolor=light_teal,
            linewidth=1.5
        )
        handles.insert(1, handles.pop(sample_traj_idx))
        labels.insert(1, labels.pop(sample_traj_idx))
        handles.insert(2, handles.pop())
        labels.insert(2, labels.pop())
        
        # leg = ax.legend(
        #     handles=handles,
        #     labels=labels,
        #     framealpha=1.0,
        #     edgecolor="0.3",
        #     loc="lower right",
        #     fontsize=24,
        #     ncol=4,
        #     handlelength=1.0,
        #     columnspacing=1.0,
        #     # bbox_to_anchor=(1.01, 1.0)
        # )
        leg = ax.legend(
            handles=handles,
            labels=labels,
            framealpha=1.0,
            edgecolor="0.3",
            loc="upper left",
            fontsize=20,
            ncol=4,
            handlelength=1.0,
            columnspacing=1.0,
            bbox_to_anchor=(-0.11, 1.18)
        )
        leg.set_zorder(12)

    if not plot_u:
        fig, ax = plt.subplots(figsize=(18, 8))
        plot_trajectory_helper(ax, T_state_gt, lows, highs, T_state_unfiltered=T_state_unfiltered, plot_cbar=plot_cbar)
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

        plot_trajectory_helper(ax_top, T_state_gt, lows, highs, T_state_unfiltered, cax=cax_top, plot_cbar=plot_cbar)

        ax_bottom.plot(T_state_gt[:, 1], T_rudder_filtered, color=pink, label="Rudder (filtered)", linewidth=1.5)
        ax_bottom.plot(T_state_gt[:, 1], T_rudder_unfiltered, color=dark_teal, label="Rudder (unfiltered)", linewidth=1.5)
        ax_bottom.set_ylabel("Rudder (degrees)", rotation=0, ha="right")
        ax_bottom.set_xlabel("DTP (m)")
        ax_bottom.legend(loc="lower left")
        ax_bottom.grid(True)
    # plt.tight_layout()
    plt.show()
    # plt.close(fig)


def plot_value_function(data, settings, frame_index=None, show=True):    
    if frame_index is None:
        x_min, x_max = -15, 15
        frame_index = 54
    else:
        x_min, x_max = -15, 15
    th_min, th_max = -np.pi/4, np.pi/4

    lows = np.array([bound.lo for bound in data.get('T_state_bounds', [])])
    highs = np.array([bound.hi for bound in data.get('T_state_bounds', [])])
    if lows.size and lows.shape[1] > 1:
        lows[:, 1] = np.rad2deg(lows[:, 1])
        highs[:, 1] = np.rad2deg(highs[:, 1])

    value_function = data['value_function']
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array([-18., -np.pi/3]), np.array([18., np.pi/3])),
        (100, 100),
    )

    x = grid.coordinate_vectors[0]
    th = grid.coordinate_vectors[1]
    ix = np.where((x >= x_min) & (x <= x_max))[0]
    ith = np.where((th >= th_min) & (th <= th_max))[0]

    # sliced versions for plotting only
    x_plot = x[ix]
    th_plot_deg = np.rad2deg(th[ith])
    vf_plot = value_function[ix][:, ith]

    # colors = [black, purple, dark_orange]
    colors = [black, purple, orange]
    cmap = mcolors.LinearSegmentedColormap.from_list("purple2orange", colors)

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(
        x_plot,
        th_plot_deg,
        vf_plot.T,
        levels=20, cmap=cmap)
    plt.colorbar(cs, ax=ax, label=r"$V(\mathbf{x})$", pad=0.01)
    ax.contour(
        x_plot,
        th_plot_deg,
        vf_plot.T,
        levels=0, colors='black', linewidths=3)
    ax.set_xlabel("CTE (m)")
    ax.set_ylabel("HE (degrees)")

    CTE_min, HE_min = lows[frame_index]
    CTE_max, HE_max = highs[frame_index]
    width = CTE_max - CTE_min
    height = HE_max - HE_min

    rect = plt.Rectangle(
        (CTE_min, HE_min),
        width, height,
        linewidth=2,
        edgecolor=light_teal,
        facecolor=light_teal + [0.4],
        linestyle='-',
        label='state_bounds',
        zorder=5)
    ax.add_patch(rect)

    if show:
        plt.show()

    return fig, ax, rect


def plot_images(data, settings):
    frame = 10
    T_image_clean = data['T_image_clean']
    T_image_est = data['T_image_est']

    img_clean = T_image_clean[frame]
    img_est = T_image_est[frame]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(
        np.asarray(np.array(img_clean, dtype=float).reshape(8, 16)), 
        cmap='gray',
        vmin=0,
        vmax=1
    )
    # axes[0].set_title("Clean Image")
    axes[1].imshow(
        np.asarray(np.array(img_est, dtype=float).reshape(8, 16)),
        cmap='gray',
        vmin=0,
        vmax=1
    )
    # axes[1].set_title("Attacked Image")
    axes[0].axis('off')
    axes[1].axis('off')
    # add vertical spacing between the two axes
    fig.subplots_adjust(hspace=0.4)
    print("Max difference:", np.max(np.abs(np.array(img_clean) - np.array(img_est))))
    print("Max attacked image value:", np.max(np.array(img_est)))
    print("Min attacked image value:", np.min(np.array(img_est)))
    print("Max clean image value:", np.max(np.array(img_clean)))
    print("Min clean image value:", np.min(np.array(img_clean)))
    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results.")
    parser.add_argument("--file", help="Path to the data file to load")
    parser.add_argument("--aux_file", help="Path to the auxiliary data file to load", default=None)
    parser.add_argument("--plot_type", help="Type of plot to generate", choices=["states", "trajectory", "value_function", "images"], default="states")
    parser.add_argument("--plot_u", action="store_true", default=False, help="Whether to plot control inputs")
    parser.add_argument("--cbar", action="store_true", default=False, help="Whether to plot colorbar")
    args = parser.parse_args()

    plot_sim(args)
