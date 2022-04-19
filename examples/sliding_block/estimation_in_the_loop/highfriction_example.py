"""
Luke Drnach
April 13, 2022
"""

import os
from pycito.systems.block.block import Block
import estimation_control_tools as campctools
import pycito.systems.terrain as terrain
import pycito.utilities as utils

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
    campctools.plot_terrain_errors(campc_controller, savedir=TARGET)
    campctools.save_estimated_terrain(campc_controller, savedir=TARGET)
    estimated_model = campctools.get_contact_model(campc_controller)
    pts = campctools.get_x_samples(campc_sim, sampling=1000)
    campctools.compare_estimated_contact_model(estimated_model, true_plant.terrain, pts, savedir=TARGET)
    campctools.compare_forces(campc_sim, campc_controller, savedir=TARGET)
    # Plot the mpc and campc logs
    campctools.plot_mpc_logs(mpc_controller, savedir = os.path.join(TARGET, 'mpc_logs'))
    campctools.save_mpc_logs(mpc_controller, savedir = os.path.join(TARGET, 'mpc_logs'))
    campctools.plot_campc_logs(campc_controller, savedir=os.path.join(TARGET,'campc_logs'))
    campctools.save_campc_logs(campc_controller, savedir=os.path.join(TARGET, 'campc_logs'))
    # Save the simulation data
    utils.save(os.path.join(TARGET, 'mpcsim.pkl'), mpc_sim)
    utils.save(os.path.join(TARGET, 'campcsim.pkl'), campc_sim)
    # Run the ambiguity optimization
    ambi_model = campctools.run_ambiguity_optimization(campc_controller.getContactEstimationTrajectory())
    campctools.compare_estimated_contact_model(ambi_model, true_plant.terrain, pts, savedir=TARGET, name='contactmodelwithambiguity')
    # Save the contact model
    utils.save(os.path.join(TARGET, 'contactambiguity.pkl'), ambi_model)

if __name__ == '__main__':
    main()