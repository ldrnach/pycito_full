"""
Luke Drnach
April 13, 2022
"""

import os
from pycito.systems.block.block import Block
import pycito.utilities as utils
import estimation_control_tools as campctools

SIM_DURATION = 1.5
TARGET = os.path.join('examples','sliding_block','estimation_in_the_loop','flatterrain')

def make_flatterrain_model():
    block = Block()
    block.Finalize()
    return block

def main():
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)
    # Make the plant and controller models
    true_plant = make_flatterrain_model()
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

def main_ambiguity():
    plant = make_flatterrain_model()
    traj = campctools.load_estimation_trajectory(TARGET)
    model = campctools.run_ambiguity_optimization(traj)
    data = utils.load(os.path.join(TARGET, 'campcsim.pkl'))
    pts = campctools.get_x_samples(data, sampling=1000)
    campctools.compare_estimated_contact_model(model, plant.terrain, pts, savedir=TARGET, name='contactmodelwithambiguity')
    # Save the contact model
    utils.save(os.path.join(TARGET, 'contactambiguity.pkl'), model)

if __name__ == '__main__':
    main_ambiguity()