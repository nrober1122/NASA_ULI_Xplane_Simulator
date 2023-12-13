import os
import pathlib
import pickle
import time

import ipdb
import numpy as np
import typer
from loguru import logger
from run_sim_lib import simulate_controller

import xpc3
import xpc3_helper
from simulation.controllers import getProportionalControl
from simulation.static_atk import StaticAttack


def run(client: xpc3.XPlaneConnect, stride: int):
    # Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
    TIME_OF_DAY = 8.0

    # Cloud cover (higher numbers are cloudier/darker)
    # 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast
    CLOUD_COVER = 0

    START_CTE = 6.0
    START_HE = 0.0
    START_DTP = 322.0
    END_DTP = 522.0

    cfg = dict(
        startCTE=START_CTE,
        startHE=START_HE,
        startDTP=START_DTP,
        endDTP=END_DTP,
        get_control=getProportionalControl,
    )

    with_images = False
    logger.info("with_images: {}".format(with_images))

    #####################################################################
    linfnorms = np.linspace(0.0, 0.035, num=10)

    logger.info("=====================")
    logger.info("Running for stride={}".format(stride))
    logger.info("=====================")
    attack = StaticAttack(stride=stride)
    results_stride = []

    for linfnorm in linfnorms[:5]:
        logger.info("linfnorm: {}".format(linfnorm))
        # Set weather and time of day
        client.sendDREF("sim/time/zulu_time_sec", TIME_OF_DAY * 3600 + 8 * 3600)
        client.sendDREF("sim/weather/cloud_type[0]", CLOUD_COVER)
        # logger.info("Waiting for reset...")
        # client.pauseSim(False)
        # client.sendDREF("sim/operation/reset_flight", 1)
        # time.sleep(2)
        # logger.info("Waiting for reset... reset done?")

        data = simulate_controller(client, attack, linfnorm, **cfg)

        if not with_images:
            # Remove the images.
            data = data.without_images()
        results_stride.append(data)

    # Save results.
    if with_images:
        results_pkl = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"]) / f"scratch/stride_results/data_{stride}_im.pkl"
    else:
        results_pkl = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"]) / f"scratch/stride_results/data_{stride}.pkl"
    results_pkl.parent.mkdir(exist_ok=True, parents=True)

    logger.info("Saving to {}...".format(results_pkl))
    with open(results_pkl, "wb") as f:
        pickle.dump(results_stride, f)
    logger.info("Saving to {}... done!".format(results_pkl))


def main(stride: int):
    with xpc3.XPlaneConnect() as client:
        run(client, stride)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
