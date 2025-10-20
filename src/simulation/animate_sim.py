import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib.animation import FuncAnimation
import pickle, json
from plot_sim import plot_value_function, data_dir
import pypalettes
from matplotlib.collections import LineCollection

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


def animate_value_function(args):
    filename = args.file
    len_history = args.len_history
    frame = args.frame
    aux_filename = args.aux_file

    # --- Load data ---
    with open(data_dir + filename + '/sim2_results.pkl', "rb") as f:
        data = pickle.load(f)
    with open(data_dir + filename + '/settings.json') as f:
        settings = json.load(f)
    # --- Initialize plot ---
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

    frame_text = fig.text(0.85, 0.05, "Frame 0", fontsize=12, ha='right', va='bottom')

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
                alpha = 0.05
            c = list(r.get_facecolor()[0:3]) + [alpha]
            r.set_facecolor(c)

        frame_text.set_text(f"Frame {frame}/{len(bounds)}")

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
        return [rect, trail_gt_lc, trail_est_lc, frame_text, arrow_unf, arrow_fil, trail_unf_lc] + trail_rects

    if frame is None:
        anim = FuncAnimation(fig, update, frames=len(bounds), interval=200, blit=False)
        plt.show()
    else:
        update(frame)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results.")
    parser.add_argument("--file", help="Path to the data file to load")
    parser.add_argument("--aux_file", help="Path to the auxiliary data file to load", default=None)
    parser.add_argument("--len_history", type=int, default=5, help="Length of the history to consider")
    parser.add_argument("--frame", type=int, default=None, help="Frame index to plot (if not animating)")
    args = parser.parse_args()

    animate_value_function(args)
