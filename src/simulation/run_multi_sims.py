import os
import sys
from datetime import datetime

xpc3_dir = os.environ["NASA_ULI_ROOT_DIR"] + "/src/"
results_dir = os.environ["NASA_ULI_DATA_DIR"] + "/logs/LOG_" + datetime.now().strftime("%Y%m%d_%H_%M_%S") + "/"
os.makedirs(results_dir, exist_ok=True)

sys.path.append(xpc3_dir)

import time
from typing import Callable

import ipdb
import time
import matplotlib.pyplot as plt
import numpy as np
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import settings
# import static_atk
# import tiny_taxinet
# import tiny_taxinet2
from loguru import logger
# from tiny_taxinet import process_image
from xplane_screenshot import get_xplane_image
from PIL import Image
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp


import xpc3
import xpc3_helper

from utils.torch2jax import torch2jax
import hj_reachability as hj
from hjnnv import hjnnvUncertaintyAwareFilter
from utils.attacks import fgsm, pgd
import dynamic_models
import tiny_taxinet2
import json

import matplotlib.pyplot as plt
from simulators.NASA_ULI_Xplane_Simulator.src.simulation.run_sim2 import simulate_controller_dubins, get_state, save_results

if __name__ == "__main__":
    num_sims = 20
    with ipdb.launch_ipdb_on_exception():
        with xpc3.XPlaneConnect() as client:
            for sim_num in range(num_sims):  # Run multiple simulations
                if sim_num == 0:
                    results_dict = None

                # Set weather and time of day
                client.sendDREF("sim/time/zulu_time_sec", settings.TIME_OF_DAY * 3600 + 8 * 3600)
                client.sendDREF("sim/weather/cloud_type[0]", settings.CLOUD_COVER)

                # Randomize START_CTE and START_HE
                START_CTE = np.random.uniform(-8, 8)
                START_HE = np.random.uniform(-20, 20)

                # Run the simulation
                results_dict = simulate_controller_dubins(
                            client,
                            START_CTE,
                            START_HE,
                            settings.START_DTP,
                            settings.END_DTP,
                            get_state,
                            settings.GET_CONTROL,
                            settings.DT,
                            settings.CTRL_EVERY,
                            results_dict=results_dict
                        )
                
            save_results(results_dict)
                

