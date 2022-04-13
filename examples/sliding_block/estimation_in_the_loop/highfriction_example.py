"""
Luke Drnach
April 13, 2022
"""

import os
from pycito.systems.block.block import Block
import estimation_control_tools as campctools
import pycito.systems.terrain as terrain

SIM_DURATION = 1.5
TARGET = os.path.join('examples','sliding_block','estimation_in_the_loop','high_friction')

def high_friction(x):
    if x[0] < 2.0 or x[0] > 4.0:
        return 0.5
    else:
        return 0.9

def make_highfriction_model():
    highfric = terrain.VariableFrictionFlatTerrain(height = 0, fric_func = high_friction)
    block = Block(terrain = highfric)
    block.Finalize()
    return block

def main():
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)
    # Make the plant and controller models
    true_plant = make_highfriction_model()
    mpc_controller = campctools.make_mpc_controller()
    mpc_controller.enableLogging()
    campc_controller = campctools.make_estimator_controller()
    campc_controller.enableLogging()
    # Run the simulation
    initial_state = mpc_controller.lintraj.getState(0)
    mpc_sim = campctools.run_simulation(true_plant, mpc_controller, initial_state, duration=SIM_DURATION)
    campc_sim = campctools.run_simulation(true_plant, campc_controller, initial_state, duration=SIM_DURATION)
    # Plot and save the results
    campctools.plot_trajectory_comparison(mpc_sim, campc_sim, savename=TARGET)
    # Plot the estimated contact model
    campctools.plot_estimated_terrain(campc_controller, savedir=TARGET)
    campctools.save_estimated_terrain(campc_controller, savedir=TARGET)
    # Plot the mpc and campc logs
    campctools.plot_mpc_logs(mpc_controller, savedir = os.path.join(TARGET, 'mpc_logs'))
    campctools.save_mpc_logs(mpc_controller, savedir = os.path.join(TARGET, 'mpc_logs'))
    campctools.plot_campc_logs(campc_controller, savedir=os.path.join(TARGET,'campc_logs'))
    campctools.save_campc_logs(campc_controller, savedir=os.path.join(TARGET, 'campc_logs'))


if __name__ == '__main__':
    main()