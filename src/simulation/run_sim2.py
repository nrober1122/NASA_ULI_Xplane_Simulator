import os
import sys

xpc3_dir = os.environ["NASA_ULI_ROOT_DIR"] + "/src/"
sys.path.append(xpc3_dir)

import time
from typing import Callable

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import settings
import static_atk
import tiny_taxinet
import tiny_taxinet2
from loguru import logger
from tiny_taxinet import process_image
from xplane_screenshot import get_xplane_image

import xpc3
import xpc3_helper


def get_state(image_raw: np.ndarray):
    image_processed = process_image(image_raw)
    image_processed += static_atk.get_patch(image_processed)
    image_processed = image_processed.clip(0, 1)

    # cte, he = tiny_taxinet.evaluate_network(image_processed)
    cte, he = tiny_taxinet2.evaluate_network(image_processed)

    return cte, he, image_processed


def get_state_clean(image_raw: np.ndarray):
    image_processed = process_image(image_raw)
    cte, he = tiny_taxinet2.evaluate_network(image_processed)
    return cte, he, image_processed


def main():
    with xpc3.XPlaneConnect() as client:
        # Set weather and time of day
        client.sendDREF("sim/time/zulu_time_sec", settings.TIME_OF_DAY * 3600 + 8 * 3600)
        client.sendDREF("sim/weather/cloud_type[0]", settings.CLOUD_COVER)

        simulate_controller(
            client,
            settings.START_CTE,
            settings.START_HE,
            settings.START_DTP,
            settings.END_DTP,
            get_state,
            settings.GET_CONTROL,
        )


def simulate_controller(
    client: xpc3.XPlaneConnect,
    startCTE: float,
    startHE: float,
    startDTP: float,
    endDTP: float,
    get_state: Callable,
    get_control: Callable,
    sim_speed: float = 1.0,
):
    client.pauseSim(True)

    client.sendDREF("sim/time/sim_speed", sim_speed)
    xpc3_helper.reset(client, cteInit=startCTE, heInit=startHE, dtpInit=startDTP)
    xpc3_helper.sendBrake(client, 0)

    client.pauseSim(False)

    dtp = startDTP
    startTime = client.getDREF("sim/time/zulu_time_sec")[0]
    now = startTime
    endTime = startTime
    run_end_time = now + 30.0

    T_state_gt, T_state_clean, T_state_est = [], [], []
    T_image_raw = []
    T_image_clean, T_image_est = [], []
    T_rudder_clean = []
    T_rudder = []

    cte_gt, dtp_gt, he_gt = xpc3_helper.getHomeState(client)

    dt = 1.0
    # dt = 0.1

    while dtp < endDTP and now < run_end_time and np.abs(cte_gt) < 10.5:
        speed = xpc3_helper.getSpeed(client)
        throttle = 0.1
        # Note: Why bang-bang controller?
        if speed > 5:
            throttle = 0.0
        elif speed < 3:
            throttle = 0.2

        cte_gt, dtp_gt, he_gt = xpc3_helper.getHomeState(client)
        state_gt = np.array([cte_gt, dtp_gt, he_gt])
        T_state_gt.append(state_gt)

        image_raw = get_xplane_image()
        T_image_raw.append(image_raw)

        cte, he, img = get_state(image_raw)
        logger.info("CTE: {:.2f} ({:.2f}), HE: {:.2f} ({:.2f})".format(cte, cte_gt, he, he_gt))
        rudder = get_control(client, cte, he)
        client.sendCTRL([0, rudder, rudder, throttle])

        cte_clean, he_clean, img_clean = get_state_clean(image_raw)
        rudder_clean = get_control(client, cte_clean, he_clean)
        T_rudder_clean.append(rudder_clean)
        T_rudder.append(rudder)

        T_image_clean.append(img_clean)
        T_image_est.append(img)

        state_est = np.array([cte, dtp_gt, he])
        T_state_est.append(state_est)
        state_clean = np.array([cte_clean, dtp_gt, he_clean])
        T_state_clean.append(state_clean)

        # Wait for next timestep. 1 Hz control rate?
        while endTime - startTime < dt:
            endTime = client.getDREF("sim/time/zulu_time_sec")[0]
            time.sleep(0.001)

        # Set things for next round
        startTime = client.getDREF("sim/time/zulu_time_sec")[0]
        endTime = startTime
        _, dtp, _ = xpc3_helper.getHomeState(client)
        time.sleep(0.001)
        now = startTime

    client.pauseSim(True)

    T_t = np.arange(len(T_state_gt)) * dt

    T_state_gt = np.stack(T_state_gt, axis=0)
    T_state_clean = np.stack(T_state_clean, axis=0)
    T_state_est = np.stack(T_state_est, axis=0)

    T_image_raw = np.stack(T_image_raw, axis=0)
    T_image_clean = np.stack(T_image_clean, axis=0)
    T_image_est = np.stack(T_image_est, axis=0)

    labels = ["CTE (m)", "DTP (m)", "HE (degrees)"]
    # Plot.
    fig, axes = plt.subplots(3, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_state_gt[:, ii], color="C4", label="True")
        ax.plot(T_t, T_state_est[:, ii], color="C1", label="Estimated")
        ax.set_ylabel(labels[ii], rotation=0, ha="right")
    axes[0].legend()
    fig.savefig("sim2_traj.pdf")
    plt.close(fig)
    ###############################################3
    # x axis = DTP, y axis = CTE.
    cte_constr = 10.0
    ylim = 11.0
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_state_gt[:, 1], T_state_gt[:, 0], marker="o", lw=0.5, color="C1")
    ax.set_aspect("equal")
    ymin, ymax = ax.get_ylim()
    ymin, ymax = min(ymin, -ylim), max(ymax, ylim)
    ax.set_ylim(ymin, ymax)
    ax.axhspan(cte_constr, ymax, color="C0", alpha=0.2)
    ax.axhspan(-cte_constr, ymin, color="C0", alpha=0.2)
    fig.savefig("sim2_plot2d.pdf")
    plt.close(fig)

    # Save the data.
    np.savez(
        "sim2_data.npz",
        T_state_gt=T_state_gt,
        T_state_clean=T_state_clean,
        T_state_est=T_state_est,
        T_image_raw=T_image_raw,
        T_image_clean=T_image_clean,
        T_image_est=T_image_est,
        T_rudder_clean=T_rudder_clean,
        T_rudder=T_rudder,
    )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
