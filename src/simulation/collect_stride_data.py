import os
import pathlib
import pickle

import ipdb
import numpy as np
from loguru import logger
from run_sim_lib import simulate_controller

import xpc3
import xpc3_helper
from simulation.controllers import getProportionalControl
from simulation.static_atk import StaticAttack


def run(client: xpc3.XPlaneConnect):
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
        startCTE=START_CTE, startHE=START_HE, startDTP=START_DTP, endDTP=END_DTP, get_control=getProportionalControl
    )

    # Set weather and time of day
    client.sendDREF("sim/time/zulu_time_sec", TIME_OF_DAY * 3600 + 8 * 3600)
    client.sendDREF("sim/weather/cloud_type[0]", CLOUD_COVER)

    #####################################################################
    results_list = []
    linfnorms = np.linspace(0.0, 0.035, num=10)
    for stride in [1, 2, 4, 8, 16]:
        logger.info("Running for stride={}".format(stride))
        attack = StaticAttack(stride=stride)
        results_stride = []

        for linfnorm in linfnorms:
            data = simulate_controller(client, attack, linfnorm, **cfg)
            results_stride.append(data)

        results_list = results_stride

    # Save results.
    results_pkl = pathlib.Path(os.environ["NASA_ULI_ROOT_DIR"]) / "scratch/stride_results/data.pkl"
    results_pkl.parent.mkdir(exist_ok=True, parents=True)
    with open(results_pkl, "wb") as f:
        pickle.dump(results_list, f)


def main():
    with xpc3.XPlaneConnect() as client:
        run(client)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
