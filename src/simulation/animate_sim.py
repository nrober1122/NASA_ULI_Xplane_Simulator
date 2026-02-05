import matplotlib.pyplot as plt
import argparse
import numpy as np
import jax.numpy as jnp
from matplotlib.animation import FuncAnimation
import pickle, json
from plot_sim import plot_value_function, data_dir
import pypalettes
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime

from dynamic_models.taxi_net import TaxiNetDynamics
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


def animate_sim(args):
    data_file = data_dir + args.file + '/sim2_results.pkl'
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    settings_file = data_dir + args.file + '/settings.json'
    with open(settings_file) as f:
        settings = json.load(f)

    if args.plot_type == "value_function":
        animate_value_function(
            data,
            settings,
            len_history=args.len_history,
            frame=args.frame,
            aux_filename=args.aux_file,
            save_animation=args.save,
            filename=args.file
            )
    elif args.plot_type == "runway_simulation":
        with open(data_dir + args.aux_file + '/sim2_results.pkl', "rb") as f:
            aux_data = pickle.load(f)
        animate_runway_simulation(
            data,
            settings,
            aux_data,
            save_animation=args.save,
            filename=args.file,
            plot_samples=args.plot_samples,
            )
    else:
        raise ValueError("Invalid plot type. Choose 'states', 'trajectory', 'value_function', or 'images'.")


def animate_runway_simulation(
    data,
    settings,
    aux_data=None,
    save_animation=False,
    filename="test_animation",
    plot_samples=False,
):
    # ============================
    # Main data
    # ============================
    T_state_gt = np.stack(data["T_state_gt"], axis=0)
    T_image_clean = data["T_image_clean"]
    T_image_est = data["T_image_est"]
    T_state_bounds = data["T_state_bounds"]
    T_rudder_filtered = data["T_rudder_filtered"]

    lows = np.array([b.lo for b in T_state_bounds])
    highs = np.array([b.hi for b in T_state_bounds])

    T_main = len(T_image_clean)

    # ============================
    # Aux data (optional)
    # ============================
    has_aux = aux_data is not None
    if has_aux:
        T_state_gt_aux = np.stack(aux_data["T_state_gt"], axis=0)
        T_image_clean_aux = aux_data["T_image_clean"]
        T_image_est_aux = aux_data["T_image_est"]
        T_state_bounds_aux = aux_data["T_state_bounds"]

        lows_aux = np.array([b.lo for b in T_state_bounds_aux])
        highs_aux = np.array([b.hi for b in T_state_bounds_aux])

        T_aux = len(T_image_clean_aux)

    if plot_samples:
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

        max_plot_samples = int(1e4)  # << tune this
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(
            num_samples,
            size=min(max_plot_samples, num_samples),
            replace=False
        )

    img_shape = (8, 16)
    title_pad = 10
    axes_fontsize = 22
    title_fontsize = 24
    ticks_fontsize = 20

    # ============================
    # Figure layout
    # ============================
    if has_aux:
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(
            5, 2,
            width_ratios=[1, 3],
            height_ratios=[1, 1, 0.35, 1, 1],
            wspace=0.40,
            hspace=0.35,
        )

        # Aux (top)
        ax_img_clean_aux = fig.add_subplot(gs[0, 0])
        ax_img_est_aux   = fig.add_subplot(gs[1, 0])
        ax_traj_aux      = fig.add_subplot(gs[0:2, 1])


        # Main (bottom)
        ax_img_clean = fig.add_subplot(gs[3, 0])
        ax_img_est   = fig.add_subplot(gs[4, 0])
        ax_traj      = fig.add_subplot(gs[3:5, 1])


    else:
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[1, 3],
            height_ratios=[1, 1],
            wspace=0.40,
            hspace=0.15,
        )

        ax_img_clean = fig.add_subplot(gs[0, 0])
        ax_img_est   = fig.add_subplot(gs[1, 0])
        ax_traj      = fig.add_subplot(gs[:, 1])

    sample_lines = []

    if plot_samples:
        for _ in range(len(sample_idx)):
            line, = ax_traj.plot(
                [],
                [],
                color=light_teal,
                alpha=0.08,
                linewidth=0.6,
                zorder=0,
            )
            sample_lines.append(line)

    # ============================
    # Helper to initialize image axes
    # ============================
    def init_image_axes(ax, title, image):
        im = ax.imshow(
            np.asarray(image).reshape(img_shape),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax.set_title(title, pad=title_pad, fontsize=title_fontsize)
        ax.axis("off")
        return im

    # ============================
    # Helper to initialize trajectory axes
    # ============================
    def init_traj_axes(ax, title, traj_color="teal"):
        ax.set_title(title, pad=title_pad, fontsize=title_fontsize)
        ax.set_xlabel("DTP (m)", fontsize=axes_fontsize)
        ax.set_ylabel("CTE (m)", fontsize=axes_fontsize)
        ax.set_ylim(-12, 12)
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
        ax.set_aspect("equal")
        ax.yaxis.set_label_coords(-0.08, 0.5)

        cte_constr = 10.0
        ax.axhline(cte_constr, color="black", linestyle="-.")
        ax.axhline(-cte_constr, color="black", linestyle="-.")
        ax.axhspan(cte_constr, 12, color="orange", alpha=0.3)
        ax.axhspan(-12, -cte_constr, color="orange", alpha=0.3)

        traj_line, = ax.plot([], [], color=traj_color, linewidth=2, zorder=3)
        traj_point, = ax.plot([], [], marker="o", color=traj_color, zorder=4)

        bounds_hi, = ax.plot([], [], color=traj_color, linewidth=0.8, zorder=2)
        bounds_lo, = ax.plot([], [], color=traj_color, linewidth=0.8, zorder=2)

        return traj_line, traj_point, bounds_hi, bounds_lo

    # ============================
    # Initialize MAIN plots
    # ============================
    im_clean = init_image_axes(ax_img_clean, "Clean Image", T_image_clean[0])
    im_est   = init_image_axes(ax_img_est, "Attacked Image", T_image_est[0])

    ax_traj.set_xlim(T_state_gt[0, 1], T_state_gt[-1, 1])
    traj_line, traj_point, bounds_hi, bounds_lo = init_traj_axes(
        ax_traj, "GUARDIAN", traj_color=dark_teal
    )
    bounds_poly = None

    legend_handles = [
        Line2D(
            [0], [0],
            color=dark_teal,
            linewidth=2,
            label=r"$\mathbf{x}_t$",
        ),
        # Patch(
        #     facecolor="teal",
        #     alpha=0.2,
        #     label=r"$\bar{\mathcal{X}}_t$",
        # ),
        Patch(
            facecolor=light_teal,
            alpha=1.0,
            label=r"${\mathcal{X}}_t$",
        ),
    ]
    ax_traj.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=18,
        frameon=True,
        ncol=2,
        framealpha=1.0,
        bbox_to_anchor=(1.01, -0.035)
    )

    # ============================
    # Initialize AUX plots (if any)
    # ============================
    if has_aux:
        im_clean_aux = init_image_axes(
            ax_img_clean_aux, "Clean Image", T_image_clean_aux[0]
        )
        im_est_aux = init_image_axes(
            ax_img_est_aux, "Attacked Image", T_image_est_aux[0]
        )

        ax_traj_aux.set_xlim(
            T_state_gt_aux[0, 1], T_state_gt_aux[-1, 1]
        )
        traj_line_aux, traj_point_aux, bounds_hi_aux, bounds_lo_aux = init_traj_axes(
            ax_traj_aux, "No Filter", traj_color=purple
        )
        bounds_poly_aux = None

    ax_traj_aux.set_xlim(ax_traj.get_xlim())
    ax_traj_aux.set_ylim(ax_traj.get_ylim())

    # ============================
    # Animation update
    # ============================
    def update(frame):
        nonlocal bounds_poly
        if has_aux:
            nonlocal bounds_poly_aux

        # -------- MAIN --------
        traj_line.set_data(
            T_state_gt[:frame + 1, 1],
            T_state_gt[:frame + 1, 0],
        )
        traj_point.set_data(
            [T_state_gt[frame, 1]],
            [T_state_gt[frame, 0]],
        )

        if plot_samples:
            for k, idx in enumerate(sample_idx):
                sample_lines[k].set_data(
                    T_state_gt[:frame + 1, 1],          # DTP
                    sample_states[:frame + 1, idx, 0],  # CTE
                )

        if bounds_poly is not None:
            bounds_poly.remove()
        # bounds_poly = ax_traj.fill_between(
        #     T_state_gt[:frame + 1, 1],
        #     lows[:frame + 1, 0],
        #     highs[:frame + 1, 0],
        #     color="teal",
        #     alpha=0.2,
        #     zorder=1,
        # )
        # bounds_hi.set_data(T_state_gt[:frame + 1, 1], highs[:frame + 1, 0])
        # bounds_lo.set_data(T_state_gt[:frame + 1, 1], lows[:frame + 1, 0])

        im_clean.set_data(np.asarray(T_image_clean[frame]).reshape(img_shape))
        im_est.set_data(np.asarray(T_image_est[frame]).reshape(img_shape))

        artists = [
            traj_line, traj_point, bounds_hi, bounds_lo, im_clean, im_est
        ]

        # -------- AUX --------
        if has_aux:
            f_aux = min(frame, T_aux - 1)

            traj_line_aux.set_data(
                T_state_gt_aux[:f_aux + 1, 1],
                T_state_gt_aux[:f_aux + 1, 0],
            )
            traj_point_aux.set_data(
                [T_state_gt_aux[f_aux, 1]],
                [T_state_gt_aux[f_aux, 0]],
            )

            # if bounds_poly_aux is not None:
            #     bounds_poly_aux.remove()
            # bounds_poly_aux = ax_traj_aux.fill_between(
            #     T_state_gt_aux[:f_aux + 1, 1],
            #     lows_aux[:f_aux + 1, 0],
            #     highs_aux[:f_aux + 1, 0],
            #     color="teal",
            #     alpha=0.2,
            #     zorder=1,
            # )
            # bounds_hi_aux.set_data(
            #     T_state_gt_aux[:f_aux + 1, 1], highs_aux[:f_aux + 1, 0]
            # )
            # bounds_lo_aux.set_data(
            #     T_state_gt_aux[:f_aux + 1, 1], lows_aux[:f_aux + 1, 0]
            # )

            im_clean_aux.set_data(
                np.asarray(T_image_clean_aux[f_aux]).reshape(img_shape)
            )
            im_est_aux.set_data(
                np.asarray(T_image_est_aux[f_aux]).reshape(img_shape)
            )

            artists += [
                traj_line_aux, traj_point_aux,
                bounds_hi_aux, bounds_lo_aux,
                im_clean_aux, im_est_aux
            ]

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=T_main,
        interval=1000 * settings["DT"],
        blit=False,
    )

    if save_animation:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        anim.save(
            data_dir + filename + '/trajectory_animation_' + datetime_str + '.gif',
            writer='pillow',
            fps=15
        )
    else:
        plt.show()

    plt.close(fig)


# def animate_runway_simulation(
#     data,
#     settings,
#     aux_data=None,
#     save_animation=False,
#     filename="test_animation",
# ):
#     T_state_gt = np.stack(data["T_state_gt"], axis=0)
#     T_image_clean = data["T_image_clean"]
#     T_image_est = data["T_image_est"]
#     T_state_bounds = data["T_state_bounds"]

#     lows = np.array([b.lo for b in T_state_bounds])
#     highs = np.array([b.hi for b in T_state_bounds])

#     num_frames = len(T_image_clean)
#     img_shape = (8, 16)

#     # ----------------------------
#     # Figure layout (IMAGES LEFT)
#     # ----------------------------
#     fig = plt.figure(figsize=(16, 6))
#     gs = gridspec.GridSpec(
#         2, 2,
#         width_ratios=[1, 3],
#         height_ratios=[1, 1],
#         wspace=0.40,
#         hspace=0.15,
#     )

#     ax_img_clean = fig.add_subplot(gs[0, 0])
#     ax_img_est = fig.add_subplot(gs[1, 0])
#     ax_traj = fig.add_subplot(gs[:, 1])

#     # ----------------------------
#     # Images
#     # ----------------------------
#     title_pad = 14
#     im_clean = ax_img_clean.imshow(
#         np.asarray(T_image_clean[0]).reshape(img_shape),
#         cmap="gray",
#         vmin=0,
#         vmax=1,
#     )
#     ax_img_clean.set_title("Clean Image", pad=title_pad)
#     ax_img_clean.axis("off")

#     im_est = ax_img_est.imshow(
#         np.asarray(T_image_est[0]).reshape(img_shape),
#         cmap="gray",
#         vmin=0,
#         vmax=1,
#     )
#     ax_img_est.set_title("Attacked Image", pad=title_pad)
#     ax_img_est.axis("off")

#     # ----------------------------
#     # Trajectory axis
#     # ----------------------------
#     ax_traj.set_title("GUARDIAN Trajectory", pad=title_pad)
#     ax_traj.set_xlabel("DTP (m)")
#     ax_traj.set_ylabel("CTE (m)")
#     ax_traj.set_xlim(T_state_gt[0, 1], T_state_gt[-1, 1])
#     ax_traj.set_ylim(-12, 12)
#     ax_traj.set_aspect("equal")
#     ax_traj.yaxis.set_label_coords(-0.08, 0.5)

#     # Constraints (static)
#     cte_constr = 10.0
#     ax_traj.axhline(cte_constr, color="black", linestyle="-.")
#     ax_traj.axhline(-cte_constr, color="black", linestyle="-.")
#     ax_traj.axhspan(cte_constr, 12, color="orange", alpha=0.3)
#     ax_traj.axhspan(-12, -cte_constr, color="orange", alpha=0.3)

#     # Trajectory artists
#     traj_line, = ax_traj.plot([], [], color="darkslategray", linewidth=2, zorder=3)
#     traj_point, = ax_traj.plot([], [], marker="o", color="darkslategray", zorder=4)

#     # Bounds artists (animated)
#     bounds_poly = None
#     bounds_hi_line, = ax_traj.plot([], [], color="teal", linewidth=0.8, zorder=2)
#     bounds_lo_line, = ax_traj.plot([], [], color="teal", linewidth=0.8, zorder=2)

#     # ----------------------------
#     # Animation update
#     # ----------------------------
#     def update(frame):
#         nonlocal bounds_poly

#         # Trajectory
#         traj_line.set_data(
#             T_state_gt[:frame + 1, 1],
#             T_state_gt[:frame + 1, 0],
#         )
#         traj_point.set_data(
#             [T_state_gt[frame, 1]],
#             [T_state_gt[frame, 0]],
#         )

#         # Bounds
#         if bounds_poly is not None:
#             bounds_poly.remove()
#         bounds_poly = ax_traj.fill_between(
#             T_state_gt[:frame + 1, 1],
#             lows[:frame + 1, 0],
#             highs[:frame + 1, 0],
#             color="teal",
#             alpha=0.2,
#             zorder=1,
#         )

#         bounds_hi_line.set_data(
#             T_state_gt[:frame + 1, 1],
#             highs[:frame + 1, 0],
#         )
#         bounds_lo_line.set_data(
#             T_state_gt[:frame + 1, 1],
#             lows[:frame + 1, 0],
#         )

#         # Images
#         im_clean.set_data(
#             np.asarray(T_image_clean[frame]).reshape(img_shape)
#         )
#         im_est.set_data(
#             np.asarray(T_image_est[frame]).reshape(img_shape)
#         )

#         return (
#             traj_line,
#             traj_point,
#             bounds_hi_line,
#             bounds_lo_line,
#             im_clean,
#             im_est,
#         )

#     anim = FuncAnimation(
#         fig,
#         update,
#         frames=num_frames,
#         interval=1000 * settings["DT"],
#         blit=False,  # safer with fill_between
#     )

#     if save_animation:
#         datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#         anim.save(data_dir + filename + '/trajectory_animation_' + datetime_str + '.gif', writer='pillow', fps=15)
#         plt.show()
#     # anim.save(f"{filename}.mp4", writer="ffmpeg", dpi=200)
#     else:
#         plt.show()

#     plt.close(fig)


def animate_value_function(
        data,
        settings,
        len_history=5,
        frame=None,
        aux_filename=None,
        save_animation=False,
        filename="test_animation"
        ):

    fig, ax, rect = plot_value_function(data, settings, frame_index=0, show=False)
    bounds = data['T_state_bounds']
    T_state_gt = np.stack(data['T_state_gt'], axis=0)[:, [0, 2]]
    T_state_est = np.stack(data['T_state_est'], axis=0)[:, [0, 2]]
    rudder_unfiltered = np.stack(data['T_rudder_unfiltered'], axis=0)
    rudder_filtered = np.stack(data['T_rudder_filtered'], axis=0)
    # T_state_gt[:, 1] = np.rad2deg(T_state_gt[:, 1])
    # T_state_est[:, 1] = np.rad2deg(T_state_est[:, 1])

    T_state_unfiltered = None
    if aux_filename is not None:
        with open(data_dir + aux_filename + '/sim2_results.pkl', "rb") as f:
            aux_data = pickle.load(f)
        T_state_unfiltered = np.stack(aux_data['T_state_gt'], axis=0)[:, [0, 2]]

    

    lows = np.array([b.lo for b in bounds])
    highs = np.array([b.hi for b in bounds])
    lows[:, 1] = np.rad2deg(lows[:, 1])
    highs[:, 1] = np.rad2deg(highs[:, 1])

    # --- Store history of recent rectangles ---
    trail_rects = []  # newest last
    # trail_gt, = ax.plot([], [], color=black, lw=2, label='True state', zorder=10)
    # trail_est, = ax.plot([], [], color=purple, lw=2, linestyle='-', label='Estimated', zorder=10)
    # Build empty line collections for both trails
    trail_gt_lc = LineCollection([], linewidths=2, zorder=10, label='True state')
    trail_est_lc = LineCollection([], linewidths=2, linestyles='--', zorder=10, label='Estimated')
    trail_unf_lc = LineCollection([], linewidths=2, linestyles='--', zorder=10, label='Unfiltered true state')
    if T_state_unfiltered is not None:
        ax.add_collection(trail_unf_lc)

    ax.add_collection(trail_gt_lc)
    ax.add_collection(trail_est_lc)

    # frame_text = fig.text(0.85, 0.05, "Frame 0", fontsize=12, ha='right', va='bottom')
    frame_text = None

    # --- Update function ---
    def update(frame):
        # Update current rectangle
        rect.set_xy((lows[frame, 0], lows[frame, 1]))
        rect.set_width(highs[frame, 0] - lows[frame, 0])
        rect.set_height(highs[frame, 1] - lows[frame, 1])

        start = max(0, frame - len_history)
        end = frame + 1
        
        
        # Compute trail range
        start = max(0, frame - 10 * len_history)
        end = frame + 1

        # True state trail (fade with alpha)
        points_gt = np.array([T_state_gt[start:end, 0], T_state_gt[start:end, 1]]).T.reshape(-1, 1, 2)
        segments_gt = np.concatenate([points_gt[:-1], points_gt[1:]], axis=1)
        alphas_gt = np.linspace(0.1, 1.0, len(segments_gt))  # older=lighter, newer=opaque
        colors_gt = np.array([black + [a] for a in alphas_gt])  # black fading

        trail_gt_lc.set_segments(segments_gt)
        trail_gt_lc.set_color(colors_gt)

        if T_state_unfiltered is not None:
            # Unfiltered state trail (fade with alpha)
            points_unf = np.array([T_state_unfiltered[start:end, 0], T_state_unfiltered[start:end, 1]]).T.reshape(-1, 1, 2)
            segments_unf = np.concatenate([points_unf[:-1], points_unf[1:]], axis=1)
            alphas_unf = np.linspace(0.1, 1.0, len(segments_unf))
            colors_unf = np.array([purple + [a] for a in alphas_unf])  # purple fading

            trail_unf_lc.set_segments(segments_unf)
            trail_unf_lc.set_color(colors_unf)

        # Estimated state trail
        points_est = np.array([T_state_est[start:end, 0], T_state_est[start:end, 1]]).T.reshape(-1, 1, 2)
        segments_est = np.concatenate([points_est[:-1], points_est[1:]], axis=1)
        alphas_est = np.linspace(0.1, 1.0, len(segments_est))
        colors_est = np.array([[0.5, 0.1, 0.8, a] for a in alphas_est])  # purple fading

        # trail_est_lc.set_segments(segments_est)
        # trail_est_lc.set_color(colors_est)

        # alpha_trail = np.linspace(0.2, 1.0, end - start)
        # trail_gt.set_alpha(0.8)
        # trail_est.set_alpha(0.6)

        # Add new rectangle to trail (as a copy)
        new_rect = plt.Rectangle(
            (lows[frame, 0], lows[frame, 1]),
            highs[frame, 0] - lows[frame, 0],
            highs[frame, 1] - lows[frame, 1],
            linewidth=1.0,
            edgecolor=light_teal + [0.4],  # teal with alpha
            facecolor=light_teal + [0.1],  # teal with alpha
            zorder=4
        )
        ax.add_patch(new_rect)
        trail_rects.append(new_rect)

        # Keep only last N
        if len(trail_rects) > len_history:
            old = trail_rects.pop(0)
            old.remove()  # remove from figure

        # Update fading
        n = len(trail_rects)
        for i, r in enumerate(trail_rects):
            # alpha = 0.05 * (i + 1) / n  # youngest higher alpha
            # alpha = 0.2/(1+np.exp(0.5*i-1))
            alpha = r.get_facecolor()[-1]
            if i < n - 1:
                alpha = 0.15
            c = list(r.get_facecolor()[0:3]) + [alpha]
            r.set_facecolor(c)

        # frame_text.set_text(f"Frame {frame}/{len(bounds)}")

        # --- Rudder arrows ---
        x0, y0 = T_state_gt[frame, 0], T_state_gt[frame, 1]
        arrow_scale = 1.0
        dy_unf = arrow_scale * rudder_unfiltered[frame]
        dy_fil = arrow_scale * rudder_filtered[frame]

        # Remove previous arrows if they exist
        for coll in ax.collections[:]:
            if getattr(coll, "rudder_arrow", False):
                coll.remove()

        # Draw new arrows
        arrow_unf = ax.quiver(
            x0+0.2, y0, 0, dy_unf,
            color=purple, angles='xy', scale_units='xy', scale=1,
            width=0.005, zorder=20
        )
        arrow_fil = ax.quiver(
            x0-0.2, y0, 0, dy_fil,
            color=dark_teal, angles='xy', scale_units='xy', scale=1,
            width=0.005, zorder=20
        )
        # Tag them so we can find/remove them next frame
        arrow_unf.rudder_arrow = True
        arrow_fil.rudder_arrow = True

        # # --- State Samples ---
        # dyn = TaxiNetDynamics(
        #     dt=settings["DT"],
        #     max_rudder=np.tan(np.deg2rad(settings["MAX_RUDDER"]))
        # )
        # # Remove previous sample collections if they exist
        # for coll in list(ax.collections):
        #     if getattr(coll, "state_samples_unf", False):
        #         coll.remove()
        #     if getattr(coll, "state_samples", False):
        #         coll.remove()

        # # Sample uniformly inside the current rectangle (use first two dims of lows/highs)
        # num_samples = 500
        # rng = np.random.default_rng()
        # lows_rad = lows.copy()
        # highs_rad = highs.copy()
        # lows_rad[:, 1] = jnp.deg2rad(lows_rad[:, 1])
        # highs_rad[:, 1] = jnp.deg2rad(highs_rad[:, 1])
        # samples = rng.uniform(low=highs_rad[frame, :2]-jnp.array([0.5, 0.05]), high=highs_rad[frame, :2], size=(num_samples, 2))

        # next_states_unfiltered = dyn.step(
        #     jnp.array(samples),
        #     jnp.ones((num_samples, 1)) * jnp.tan(jnp.deg2rad(rudder_unfiltered[frame])),
        #     jnp.zeros((num_samples, 2))
        # )
        # next_states = dyn.step(
        #     jnp.array(samples),
        #     jnp.ones((num_samples, 1)) * jnp.tan(jnp.deg2rad(rudder_filtered[frame])),
        #     jnp.zeros((num_samples, 2))
        # )
        # # next_states = []
        # # for sample in samples:
        # #     next_state = dyn.step(
        # #         jnp.array(sample),
        # #         jnp.zeros((1,)),
        # #         jnp.zeros((2,))
        # #     )
        # #     next_states.append(next_state)
        
        # # next_states = jnp.stack(next_states, axis=0)

        # # Plot samples and tag the collection so we can remove it next frame
        # sample_scatter_unf = ax.scatter(next_states_unfiltered[:, 0], jnp.rad2deg(next_states_unfiltered[:, 1]),
        #                 s=10, color=light_teal, alpha=0.7, zorder=12)
        # sample_scatter_unf.state_samples = True
        # sample_scatter = ax.scatter(next_states[:, 0], jnp.rad2deg(next_states[:, 1]),
        #                 s=10, color=purple, alpha=0.7, zorder=12)
        # sample_scatter.state_samples = True
        # import ipdb; ipdb.set_trace()
        return [rect, trail_gt_lc, trail_est_lc, frame_text, arrow_unf, arrow_fil, trail_unf_lc] + trail_rects

    if frame is None:
        anim = FuncAnimation(fig, update, frames=len(bounds), interval=150, blit=False)
        if save_animation:
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            anim.save(data_dir + filename + '/value_animation_' + datetime_str + '.gif', writer='pillow', fps=5)
        plt.show()
    else:
        update(frame)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results.")
    parser.add_argument("--plot_type", type=str, required=True,
                        choices=["value_function", "runway_simulation"],
                        help="Type of plot to generate.")
    parser.add_argument("--file", help="Path to the data file to load")
    parser.add_argument("--aux_file", help="Path to the auxiliary data file to load", default=None)
    parser.add_argument("--len_history", type=int, default=5, help="Length of the history to consider")
    parser.add_argument("--frame", type=int, default=None, help="Frame index to plot (if not animating)")
    parser.add_argument("--plot_samples", default=False, action="store_true", help="Whether to plot sample trajectories")
    parser.add_argument("--save", default=False, action="store_true", help="Whether to save the animation as a GIF")
    args = parser.parse_args()

    # animate_value_function(args)
    animate_sim(args)
