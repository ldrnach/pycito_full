"""
General Script for making an optimization configuration for A1

Luke Drnach
July 2, 2021
"""
import os
import numpy as np
from trajopt.optimizer import A1OptimizerConfiguration

config = A1OptimizerConfiguration.defaultWalkingConfig()
allntimes = [21, 51] 
allvelcost = [10, 50, 100]
allcompcost = [10, 100, 1000]
allequalsteps = [True, False]
config.maximum_time = 2
config.solver_options['Major Iterations Limit']  =  5000
# Background material
Q, xr = config.quadratic_state_cost
# Upweight the velocities
weights = np.diag(Q).copy()
nx = weights.shape[0]
# Create the save path
dirname = os.path.join('examples','a1','runs')
# Loop over and create configurations
for equalsteps in allequalsteps:
    # Fixed time steps
    config.useFixedTimesteps = equalsteps
    for compcost in allcompcost:
        # Set complementarity cost
        config.complementarity_cost_weight = compcost
        for velcost in allvelcost:
            # Set Velocity weights
            weights[int(nx/2):] = velcost
            config.quadratic_state_cost = (np.diag(weights), xr)
            for ntimes in allntimes:
                # Set timesteps
                config.num_time_samples = ntimes
                # Create a save string
                filename = f"walking_N{ntimes}_VelCost{velcost}_ComplCost{compcost}"
                if equalsteps:
                    filename += "equaltime.pkl"
                else:
                    filename += ".pkl"
                # Save the configuration
                savename = os.path.join(dirname, filename)
                config.save(savename)
