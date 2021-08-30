"""
General Script for making an optimization configuration for the sliding block

Luke Drnach
July 2, 2021
"""

import os
from time import time
from datetime import date
import numpy as np
from trajopt.optimizer import BlockOptimizerConfiguration
from copy import deepcopy
from utilities import save

# Create the save path
dirname = os.path.join('examples','sliding_block','runs')
# Get the generic configuration
config = BlockOptimizerConfiguration.defaultBlockConfig()

# Create the random initial state samples
samples = np.random.default_rng().uniform(0, 10, (200,))

for n in range(200):
    config.initial_state[0] = samples[n]
    filename = f"block_config_sample{n}.pkl"
    savename = os.path.join(dirname, filename)
    config.save(savename)
savedir = os.path.join('examples','sliding_block','block_initial_state_samples.pkl')
save(savedir, samples)
