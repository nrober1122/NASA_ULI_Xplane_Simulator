# Defines settings for collecting training data by running sinusoidal trajectories
# Written by Sydney Katz (smkatz@stanford.edu)
import yaml
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import controllers
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import fully_observable
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import pretrained_dnn
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import tiny_taxinet
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import tiny_taxinet2
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import static_atk
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import static_atk_dnn
from simulators.NASA_ULI_Xplane_Simulator.src.train_DNN import model_taxinet
from utils.attacks import fgsm, pgd
from functools import partial
from pathlib import Path

""" 
Parameters to be specified by user
    - Change these parameters to determine the cases you want to gather data for
"""
THIS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = THIS_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# If you want to set variables from config dictionary:
DUBINS = config["DUBINS"]
STATE_ESTIMATOR = config["STATE_ESTIMATOR"]
USING_TORCH = config["USING_TORCH"]
SMOOTHING = config["SMOOTHING"]
SMOOTHING_ALPHA = config["SMOOTHING_ALPHA"]

ATTACK_STRING = config["ATTACK"]        # Will be None if YAML has null
ATTACK_STRENGTH = config["ATTACK_STRENGTH"]
TARGET = config.get("TARGET", 7.0)
PGD_ALPHA = config["PGD_ALPHA"]
PGD_STEPS = config["PGD_STEPS"]

FILTER = config["FILTER"]
CTE_BUFFER = config["CTE_BUFFER"]
HE_BUFFER = config["HE_BUFFER"]

TIME_OF_DAY = config["TIME_OF_DAY"]
CLOUD_COVER = config["CLOUD_COVER"]
START_CTE = config["START_CTE"]
START_HE = config["START_HE"]
START_DTP = config["START_DTP"]
END_DTP = config["END_DTP"]
DT = config["DT"]
MAX_RUDDER = config["MAX_RUDDER"]
CTRL_EVERY = config["CTRL_EVERY"]
IMAGE_NOISE = config["IMAGE_NOISE"]

# # Whether or not to override the X-Plane 11 simulator dynamics with a Dubin's car model
# # True = Use Dubin's car model
# # False = Use X-Plane 11 dynamics
# DUBINS = True

# # Type of state estimation
# # 'tiny_taxinet'     - state is estimated using the tiny taxinet neural network from
# #                      image observations of the true state
# # 'dnn'              - state is estimated using the resnet neural network from
# #                      image observations of the true state
# # STATE_ESTIMATOR = 'tiny_taxinet'
# STATE_ESTIMATOR = 'dnn'
# USING_TORCH = True

"""
Other parameters
    - NOTE: you should not need to change any of these unless you want to create
    additional scenarios beyond the ones provided
"""

# Tells simulator which proportional controller to use based on dynamics model
if DUBINS:
    GET_CONTROL = controllers.getProportionalControlDubins
else:
    GET_CONTROL = controllers.getProportionalControl

# Tells simulator which function to use to estimate the state
if STATE_ESTIMATOR == 'tiny_taxinet':
    PROCESS_IMG = tiny_taxinet.process_image
    GET_STATE_SMOOTHED = partial(tiny_taxinet2.evaluate_network_smoothed, alpha=SMOOTHING_ALPHA)
    GET_STATE = tiny_taxinet2.evaluate_network
    NETWORK = tiny_taxinet2.get_network
    TARGET_FUNCTION = tiny_taxinet2.target_function
    PACKAGE_INPUT = tiny_taxinet2.package_input
elif STATE_ESTIMATOR == 'fully_observable':
    GET_STATE = fully_observable.getStateFullyObservable
elif STATE_ESTIMATOR == 'dnn':
    PROCESS_IMG = pretrained_dnn.process_image
    GET_STATE = pretrained_dnn.evaluate_network
    NETWORK = pretrained_dnn.get_network
elif STATE_ESTIMATOR in ['cnn', 'cnn64']:
    if STATE_ESTIMATOR == 'cnn':
        H, W = 224, 224
        in_channels = 3
    elif STATE_ESTIMATOR == 'cnn64':
        H, W = 64, 64
        in_channels = 1
    print(f"Using {STATE_ESTIMATOR} with in_channels={in_channels}, H={H}, W={W}")
    PROCESS_IMG = partial(pretrained_dnn.process_image, in_channels=in_channels, width=W, height=H)
    GET_STATE = partial(pretrained_dnn.evaluate_network, in_channels=in_channels, width=W, height=H)
    GET_STATE_SMOOTHED = partial(
        pretrained_dnn.evaluate_network_smoothed,
        in_channels=in_channels,
        width=W,
        height=H,
        alpha=SMOOTHING_ALPHA
    )
    NETWORK = partial(pretrained_dnn.get_network, in_channels=in_channels, W=W, H=H)
    TARGET_FUNCTION = pretrained_dnn.target_function
    PACKAGE_INPUT = pretrained_dnn.package_input
    # PROCESS_IMG = pretrained_dnn.process_image
    # GET_STATE = pretrained_dnn.evaluate_network
    # NETWORK = pretrained_dnn.get_network
else:
    print("Invalid state estimator name - assuming fully observable")
    GET_STATE = fully_observable.getStateFullyObservable

if ATTACK_STRING is None:
    ATTACK_STRING = "null"
    ATTACK = None
elif ATTACK_STRING == 'static_atk':
    ATTACK = static_atk
elif ATTACK_STRING == 'static_atk_dnn':
    ATTACK = static_atk_dnn
elif ATTACK_STRING == 'fgsm':
    ATTACK = fgsm
elif ATTACK_STRING == 'pgd':
    ATTACK = pgd
else:
    ATTACK = lambda image: 0.0*image  # No attack if invalid string
