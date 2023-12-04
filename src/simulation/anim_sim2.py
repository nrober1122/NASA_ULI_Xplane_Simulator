import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def main():
    npz = np.load("sim2_data.npz")

    T_state_gt = npz["T_state_gt"]
    T_state_clean = npz["T_state_clean"]
    T_state_est = npz["T_state_est"]
    # ()
    T_image_raw = npz["T_image_raw"]
    T_image_clean = npz["T_image_clean"]
    T_image_est = npz["T_image_est"]
    T_rudder_clean = npz["T_rudder_clean"]
    T_rudder_est = npz["T_rudder"]

    # BGRA -> RGB
    T_image_raw = T_image_raw[:, :, :, [2, 1, 0]]
    T_image_clean = T_image_clean.reshape(-1, 8, 16)
    T_image_est = T_image_est.reshape(-1, 8, 16)

    figsize = 2 * np.array([3 * 2, 1.5])
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")

    im_raw = axes[0].imshow(T_image_raw[0])
    im_clean = axes[1].imshow(T_image_clean[0], cmap="gray")
    im_est = axes[2].imshow(T_image_est[0], cmap="gray")

    axes[0].set_title("Raw Image")
    axes[1].set_title("Downscaled")
    axes[2].set_title("Downscaled + Attacked")

    def get_text(cte, he, rudder):
        return "CTE: {:.2f}, HE: {:.2f}\nRudder: {:.2f}".format(cte, he, rudder)

    text_opts = dict(size=12, va="top", ha="center")

    clean_text = axes[1].text(
        0.5,
        -0.1,
        get_text(0, 1, 2),
        **text_opts,
        transform=axes[1].transAxes,
    )
    atk_text = axes[2].text(
        0.5,
        -0.1,
        get_text(0, 1, 2),
        **text_opts,
        transform=axes[2].transAxes,
    )

    [ax.axis("off") for ax in axes]
    [ax.grid(False) for ax in axes]
    fig.savefig("sim2_anim.pdf")

    def init_fn() -> list[plt.Artist]:
        return [clean_text, atk_text, im_raw, im_clean, im_est]

    def update(kk: int) -> list[plt.Artist]:
        # Update images.
        im_raw.set_data(T_image_raw[kk])
        im_clean.set_data(T_image_clean[kk])
        im_est.set_data(T_image_est[kk])

        # Update text.
        state_clean = T_state_clean[kk]
        clean_text.set_text(get_text(state_clean[0], state_clean[2], T_rudder_clean[kk]))

        state_est = T_state_est[kk]
        atk_text.set_text(get_text(state_est[0], state_est[2], T_rudder_est[kk]))

        return [clean_text, atk_text, im_raw, im_clean, im_est]

    plot_dir = pathlib.Path("plots")

    fps = 2.0
    spf = 1 / fps
    mspf = 1_000 * spf
    ani = FuncAnimation(fig, update, frames=len(T_state_gt), init_func=init_fn, interval=mspf, blit=True)
    ani_path = plot_dir / "sim2.mp4"
    ani_path.parent.mkdir(exist_ok=True)
    ani.save(ani_path)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
