# Defines settings for collecting training data by running sinusoidal trajectories
# Written by Sydney Katz (smkatz@stanford.edu)

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

# Whether or not to override the X-Plane 11 simulator dynamics with a Dubin's car model
# True = Use Dubin's car model
# False = Use X-Plane 11 dynamics
DUBINS = False

# Type of state estimation
# 'tiny_taxinet'     - state is estimated using the tiny taxinet neural network from
#                      image observations of the true state
# 'dnn'              - state is estimated using the resnet neural network from
#                      image observations of the true state
# STATE_ESTIMATOR = 'tiny_taxinet'
STATE_ESTIMATOR = 'dnn'

# Type of adversarial attack
# None                  - No adversarial attack
# static_atk.get_patch  - patch generated for tiny taxinet

# ATTACK = None
# ATTACK = static_atk
ATTACK = static_atk_dnn

# Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
TIME_OF_DAY = 8.0

# Cloud cover (higher numbers are cloudier/darker)
# 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast
CLOUD_COVER = 0

# Starting crosstrack error in meters
START_CTE = 6.0

# Starting heading error in degrees
START_HE = 0.0

# Starting downtrack position in meters
START_DTP = 322.0

# Downtrack positions (in meters) to end the simulation
END_DTP = 522.0

"""
Parameters for Dubin's Model
"""

# Time steps for the dynamics in seconds
DT = 0.1

# Frequency to get new control input 
# (e.g. if DT=0.5, CTRL_EVERY should be set to 20 to perform control at a 1 Hz rate)
CTRL_EVERY = 20

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
elif STATE_ESTIMATOR == 'fully_observable':
    GET_STATE = fully_observable.getStateFullyObservable
elif STATE_ESTIMATOR == 'dnn':
    PROCESS_IMG = pretrained_dnn.process_image
    GET_STATE = pretrained_dnn.evaluate_network
else:
    print("Invalid state estimator name - assuming fully observable")
    GET_STATE = fully_observable.getStateFullyObservable
