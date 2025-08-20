# Defines settings for collecting training data by running sinusoidal trajectories
# Written by Sydney Katz (smkatz@stanford.edu)
import yaml
import controllers
import fully_observable
import pretrained_dnn
import tiny_taxinet
import tiny_taxinet2
import static_atk
import static_atk_dnn

""" 
Parameters to be specified by user
    - Change these parameters to determine the cases you want to gather data for
"""
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# If you want to set variables from config dictionary:
DUBINS = config["DUBINS"]
STATE_ESTIMATOR = config["STATE_ESTIMATOR"]
USING_TORCH = config["USING_TORCH"]
ATTACK_STRING = config["ATTACK"]        # Will be None if YAML has null
ATTACK_STRENGTH = config["ATTACK_STRENGTH"]
FILTER = config["FILTER"]
TIME_OF_DAY = config["TIME_OF_DAY"]
CLOUD_COVER = config["CLOUD_COVER"]
START_CTE = config["START_CTE"]
START_HE = config["START_HE"]
START_DTP = config["START_DTP"]
END_DTP = config["END_DTP"]
DT = config["DT"]
CTRL_EVERY = config["CTRL_EVERY"]

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

# # Type of adversarial attack
# # None                  - No adversarial attack
# # static_atk.get_patch  - patch generated for tiny taxinet

# ATTACK = None
# # ATTACK = static_atk
# # ATTACK = static_atk_dnn
# ATTACK_STRENGTH = 0.035

# FILTER = False

# # Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
# TIME_OF_DAY = 8.0

# # Cloud cover (higher numbers are cloudier/darker)
# # 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast
# CLOUD_COVER = 0

# # Starting crosstrack error in meters
# START_CTE = 6.0

# # Starting heading error in degrees
# START_HE = 0.0

# # Starting downtrack position in meters
# START_DTP = 322.0

# # Downtrack positions (in meters) to end the simulation
# END_DTP = 522.0

# """
# Parameters for Dubin's Model
# """

# # Time steps for the dynamics in seconds
# DT = 0.1

# # Frequency to get new control input 
# # (e.g. if DT=0.05, CTRL_EVERY should be set to 20 to perform control at a 1 Hz rate)
# CTRL_EVERY = 1

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
    GET_STATE = tiny_taxinet2.evaluate_network
    NETWORK = tiny_taxinet2.get_network
elif STATE_ESTIMATOR == 'fully_observable':
    GET_STATE = fully_observable.getStateFullyObservable
elif STATE_ESTIMATOR == 'dnn':
    PROCESS_IMG = pretrained_dnn.process_image
    GET_STATE = pretrained_dnn.evaluate_network
    NETWORK = pretrained_dnn.get_network
elif STATE_ESTIMATOR == 'cnn':
    PROCESS_IMG = pretrained_dnn.process_image
    GET_STATE = pretrained_dnn.evaluate_network
    NETWORK = pretrained_dnn.get_network
else:
    print("Invalid state estimator name - assuming fully observable")
    GET_STATE = fully_observable.getStateFullyObservable

if ATTACK_STRING is None:
    ATTACK = None
elif ATTACK_STRING == 'static_atk':
    ATTACK = static_atk
elif ATTACK_STRING == 'static_atk_dnn':
    ATTACK = static_atk_dnn
