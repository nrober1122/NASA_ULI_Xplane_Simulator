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
import tiny_taxinet
import tiny_taxinet2
from loguru import logger
from tiny_taxinet import process_image
from xplane_screenshot import get_xplane_image

import xpc3
import xpc3_helper


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
            settings.GET_CONTROL,
        )


def simulate_controller(
    client: xpc3.XPlaneConnect,
    startCTE: float,
    startHE: float,
    startDTP: float,
    endDTP: float,
    get_control: Callable,
    sim_speed: float = 1.0,
):
    client.sendDREF("sim/time/sim_speed", sim_speed)
    xpc3_helper.reset(client, cteInit=startCTE, heInit=startHE, dtpInit=startDTP)
    xpc3_helper.sendBrake(client, 0)

    client.pauseSim(False)

    dtp = startDTP
    startTime = client.getDREF("sim/time/zulu_time_sec")[0]
    endTime = startTime

    T_state_gt, T_state_est = [], []

    while dtp < endDTP:
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
        image_processed = process_image(image_raw)
        # cte, he = tiny_taxinet.evaluate_network(image_processed)
        cte, he = tiny_taxinet2.evaluate_network(image_processed)
        logger.info("CTE: {:.2f}, HE: {:.2f}".format(cte, he))
        rudder = get_control(client, cte, he)
        client.sendCTRL([0, rudder, rudder, throttle])

        state_est = np.array([cte, dtp_gt, he])
        T_state_est.append(state_est)

        # Wait for next timestep. 1 Hz control rate?
        while endTime - startTime < 1:
            endTime = client.getDREF("sim/time/zulu_time_sec")[0]
            time.sleep(0.001)

        # Set things for next round
        startTime = client.getDREF("sim/time/zulu_time_sec")[0]
        endTime = startTime
        _, dtp, _ = xpc3_helper.getHomeState(client)
        time.sleep(0.001)

    client.pauseSim(True)

    dt = 1.0
    T_t = np.arange(len(T_state_gt)) * dt

    T_state_gt = np.stack(T_state_gt, axis=0)
    T_state_est = np.stack(T_state_est, axis=0)

    labels = ["CTE (m)", "DTP (m)", "HE (degrees)"]
    # Plot.
    fig, axes = plt.subplots(3, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_state_gt[:, ii], color="C4", label="True")
        ax.plot(T_t, T_state_est[:, ii], color="C1", label="Estimated")
        ax.set_ylabel(labels[ii], rotation=0, ha="right")
    axes[0].legend()
    fig.savefig("sim2_traj.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
