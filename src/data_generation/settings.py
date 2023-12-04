import os
# Defines settings for collecting training data by running sinusoidal trajectories
# Written by Sydney Katz (smkatz@stanford.edu)

""" 
Parameters to be specified by user
    - Change these parameters to determine the cases you want to gather data for
"""

# Directory to save output data
# NOTE: CSV file and images will be overwritten if already exists in that directory, but
# extra images (for time steps that do not occur in the new episodes) will not be deleted
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
OUT_DIR = NASA_ULI_ROOT_DIR + "/scratch/large_images/morning_nick/morning_nick_validation/"

# # Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
# TIME_OF_DAY = 8.0

# Start and end of range of possible time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
# For each sinsoidal trajectory in the run, the time of day will be sampled uniformly
# from this range
TIME_OF_DAY_START = 8.0
TIME_OF_DAY_END = 11.0

# Cloud cover (higher numbers are cloudier/darker)
# 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast
CLOUD_COVER = 0

# Cases to run (determines how other variables are set)
# example    - runs 2 short trajectories (used for initial testing)
# smallset   - runs 5 sinusoidal trajectories centered at zero crosstrack error with 
#              varying amplitude and frequency (ideal for collecting OoD data)
# largeset   - runs 20 sinusoidal trajectories with varying amplitude and frequency
#              and centered at different crosstrack errors
# validation - runs 5 sinusoidal trajectories centered at zero crosstrack error with 
#              varying amplitude and frequency
# test       - runs 3 sinusoidal trajectories center at 3 different crosstrack errors
# the last five trajectories of the largeset have the same parameter settings as
# the smallset
case = 'example'

# Frequency with which to record data
# NOTE: this is approximate due to computational overhead
FREQUENCY = 5 # Hz

"""
Other parameters
    - NOTE: you should not need to change any of these unless you want to create
    additional scenarios beyond the ones provided
"""

# Case indices to run (see getParams in sinusoidal.py for the specifics of each case)
if case == 'example':
    CASE_INDS = [18, 19]
elif case == 'smallset':
    CASE_INDS = [*range(15, 20)]
elif case == 'largeset':
    CASE_INDS = [*range(0, 20)]
elif case == 'validation':
    CASE_INDS = [*range(20, 25)]
elif case == 'test':
    CASE_INDS = [*range(25, 28)]
else:
    print('invalid case name, running the example set...')
    CASE_INDS = [18, 19]

# Percentage down the runway to end the trajectory (shorter for the example trajectories)
if case == 'smallset' or case == 'largeset' or case == 'validation' or case == 'test':
    END_PERC = 95.0
else:
    END_PERC = 4.0

# Screenshot parameters
MONITOR = {'top': 100, 'left': 100, 'width': 1720, 'height': 960}
# Width and height of final image
WIDTH = 360
HEIGHT = 200

# Dictionary of weather types (used when selecting a name for each output image)
WEATHER_TYPES = {0: 'clear', 1: 'cirrus', 2: 'scattered', 3: 'broken', 4: 'overcast'}
