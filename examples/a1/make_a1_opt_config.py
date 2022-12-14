"""
General Script for making an optimization configuration for A1

Luke Drnach
July 2, 2021
"""
import os
from time import time
from datetime import date
import numpy as np
from pycito.trajopt.optimizer import A1OptimizerConfiguration
from copy import deepcopy
from pycito.utilities import save

# Create the save path
dirname = os.path.join('examples','a1','runs')
datename = date.today().strftime("%b-%d-%Y")
# Get the generic configuration
config = A1OptimizerConfiguration.defaultWalkingConfig()
# New fixed settings
config.num_time_samples = 101
config.maximum_time = 1
config.minimum_time = 1
config.useFixedTimesteps = True
config.solver_options['Major iterations limit']  =  5000
# Background material
Q, xr = config.quadratic_state_cost
R, ur = config.quadratic_control_cost
# Upweight the velocities
weights = np.diag(Q).copy()
nx = weights.shape[0]
weights[int(nx/2):] = 100
config.quadratic_state_cost = (np.diag(weights), xr)
# Control weights
# Rweights = np.diag(R).copy()

# Set the slack method
config.complementarity = 'useNonlinearComplementarityWithCost'
distance_cost = [1, 10, 100, 1000, 10000]
config.complementarity_cost_weight = 1
warmstartdir = os.path.join('data','a1','warmstarts')
warmstarts = ['liftedlinear','staticwalking']
warmfiles = [os.path.join(warmstartdir, filename) for filename in warmstarts]
for file in warmstarts:
    warmfile = os.path.join(warmstartdir, file+'.pkl')
    config.initial_guess = ('useCustomGuess', warmfile)
    config_list = []
    filename = f"a1_walking_warmstart_{file}"
    for d_cost in distance_cost:
        new_config = deepcopy(config)
        new_config.complementarity_cost_weight = d_cost
        config_list.append(new_config)
    # Save the configuration
    savename = os.path.join(dirname, filename) + '.pkl'
    save(savename, config_list)






# Loop over and create configurations
# for duration in alltimes:
#     # Fixed time steps
#     config.minimum_time = duration
#     config.maximum_time = duration
#     filename = f"walking_T{duration}"
#     for heightcost  in allheightcost:
#         weights[2] = heightcost
#         config.quadratic_state_cost = (np.diag(weights), xr)
#         filename += f"_HeightCost{heightcost}"
#         for controlcost in allcontrolcost:
#             Rweights[:] = controlcost
#             filename += f"_ControlCost{controlcost}"
#             for useref in usecontrolreference:
#                 if useref:
#                     config.quadratic_control_cost = (np.diag(Rweights), ur)
#                     filename += f"_ControlRef"
#                 else:
#                     config.quadratic_control_cost = (np.diag(Rweights), np.zeros_like(ur))
#                 # Make a list of configurations for complementarity costs
#                 config_list = []
#                 for compcost in allcompcost:
#                     newconfig = deepcopy(config)
#                     newconfig.complementarity_cost_weight = compcost
#                     config_list.append(newconfig)

#                 filename = f"walking_T{duration}_Height{heightcost}_Control{controlcost:.0E}_ControlRef{useref}"
#                 # Save the configuration
#                 savename = os.path.join(dirname, filename + ".pkl")
#                 save(savename, config_list)

