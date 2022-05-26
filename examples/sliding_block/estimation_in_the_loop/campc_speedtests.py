"""
Luke Drnach
April 13, 2022
"""

import os
import numpy as np
from pycito.systems.block.block import Block
import estimation_control_tools as campctools
import pycito.systems.terrain as terrain
import pycito.systems.kernels as kernels 
import matplotlib.pyplot as plt

SIM_DURATION = 1.5
TARGET = os.path.join('examples','sliding_block','estimation_in_the_loop','paper','speedtests')
HORIZONS = range(1, 21)

def make_flatterrain_model():
    block = Block()
    block.Finalize()
    return block

def high_friction(x):
    if x[0] < 2.0 or x[0] > 4.0:
        return 0.5
    else:
        return 0.9

def make_kernel_model():
    W = np.diag([0.1, 0.1, 0.0])
    surfkernel = kernels.CompositeKernel(
        kernels.CenteredLinearKernel(W),
        kernels.ConstantKernel(1),
        kernels.WhiteNoiseKernel(0.01)
    )
    frickernel = kernels.RegularizedConstantKernel(1, 0.01)
    return campctools.make_semiparametric_contact_model(surfkernel, frickernel)

def get_global_kernel():
    return kernels.RegularizedPseudoHuberKernel(length_scale=np.array([0.1, 0.1, np.inf]), delta=0.1, noise=0.01)

def make_highfriction_model():
    highfric = terrain.VariableFrictionFlatTerrain(height = 0, fric_func = high_friction)
    block = Block(terrain = highfric)
    block.Finalize()
    return block

def make_stepterrain_model():
    stepterrain = terrain.StepTerrain(step_height = -0.5, step_location=2.5)
    block = Block(terrain = stepterrain)
    block.Finalize()
    return block

def main_flatterrain_horizons():
    for horizon in HORIZONS:
        print(f"Testing horizon {horizon} for flatterrain")
        campctools.MPC_HORIZON = horizon
        campctools.ESTIMATION_HORIZON = horizon
        campctools.run_piecewise_estimation_control(make_flatterrain_model(), 
            make_kernel_model(),
            get_global_kernel(),
            savedir = os.path.join(TARGET, 'flatterrain',f'horizon_{horizon}'))
        plt.close('all')

def main_stepterrain_horizons():
    for horizon in HORIZONS:
        print(f"Testing horizon {horizon} for stepterrain")
        campctools.MPC_HORIZON = horizon
        campctools.ESTIMATION_HORIZON = horizon
        campctools.run_piecewise_estimation_control(make_stepterrain_model(), 
            make_kernel_model(),
            get_global_kernel(),
            savedir = os.path.join(TARGET, 'stepterrain',f'horizon_{horizon}'))
        plt.close('all')

def main_highfriction_horizons():
    for horizon in HORIZONS:
        print(f"Testing horizon {horizon} for highfriction")
        campctools.MPC_HORIZON = horizon
        campctools.ESTIMATION_HORIZON = horizon
        campctools.run_piecewise_estimation_control(make_highfriction_model(), 
            make_kernel_model(),
            get_global_kernel(),
            savedir = os.path.join(TARGET, 'highfriction',f'horizon_{horizon}'))
        plt.close('all')

if __name__ == '__main__':
    main_flatterrain_horizons()
    main_stepterrain_horizons()
    main_highfriction_horizons()