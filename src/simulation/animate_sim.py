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
def animate_value_function(file, len_history=5):
    # --- Load data ---
    with open(data_dir + file + '/sim2_results.pkl', "rb") as f:
        data = pickle.load(f)
    with open(data_dir + file + '/settings.json') as f:
        settings = json.load(f)

    # --- Initialize plot ---
    fig, ax, rect = plot_value_function(data, settings, frame_index=0, show=False)
    bounds = data['T_state_bounds']
    T_state_gt = np.stack(data['T_state_gt'], axis=0)[:, [0, 2]]
    T_state_est = np.stack(data['T_state_est'], axis=0)[:, [0, 2]]
    # T_state_gt[:, 1] = np.rad2deg(T_state_gt[:, 1])
    # T_state_est[:, 1] = np.rad2deg(T_state_est[:, 1])

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
        colors_gt = np.array([[0.0, 0.0, 0.0, a] for a in alphas_gt])  # black fading

        trail_gt_lc.set_segments(segments_gt)
        trail_gt_lc.set_color(colors_gt)

        # Estimated state trail
        points_est = np.array([T_state_est[start:end, 0], T_state_est[start:end, 1]]).T.reshape(-1, 1, 2)
        segments_est = np.concatenate([points_est[:-1], points_est[1:]], axis=1)
        alphas_est = np.linspace(0.1, 1.0, len(segments_est))
        colors_est = np.array([[0.5, 0.1, 0.8, a] for a in alphas_est])  # purple fading

        trail_est_lc.set_segments(segments_est)
        trail_est_lc.set_color(colors_est)

        # alpha_trail = np.linspace(0.2, 1.0, end - start)
        # trail_gt.set_alpha(0.8)
        # trail_est.set_alpha(0.6)

        # Add new rectangle to trail (as a copy)
        new_rect = plt.Rectangle(
            (lows[frame, 0], lows[frame, 1]),
            highs[frame, 0] - lows[frame, 0],
            highs[frame, 1] - lows[frame, 1],
            linewidth=0,
            facecolor=light_teal + [0.3],  # teal with alpha
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
            alpha = 0.2
            c = list(r.get_facecolor()[0:3]) + [alpha]
            r.set_facecolor(c)

        frame_text.set_text(f"Frame {frame}/{len(bounds)}")
        return [rect, trail_gt_lc, trail_est_lc, frame_text] + trail_rects

    anim = FuncAnimation(fig, update, frames=len(bounds), interval=200, blit=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results.")
    parser.add_argument("--file", help="Path to the data file to load")
    args = parser.parse_args()

    animate_value_function(args.file, len_history=5)
