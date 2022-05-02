"""
Luke Drnach
April 13, 2022
"""
#TODO: Use the entire contact model, not just the local one
#TODO: retune the controller to get better tracking performance


import os
from pycito.systems.block.block import Block
import estimation_control_tools as campctools
import pycito.systems.terrain as terrain
import pycito.utilities as utils
import campc_animation_tools as animator
from pycito.controller.optimization import OptimizationLogger

SIM_DURATION = 1.5
TARGET = os.path.join('examples','sliding_block','estimation_in_the_loop','stepterrain')
ANIMATION_NAME = 'campc_animation.mp4'
MPCANIMATIONNAME = 'mpc_animation.mp4'

def make_stepterrain_model():
    stepterrain = terrain.StepTerrain(step_height = -0.5, step_location=2.5)
    block = Block(terrain = stepterrain)
    block.Finalize()
    return block

def main():
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)
    # Make the plant and controller models
    true_plant = make_stepterrain_model()
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
    # Save the reference trajectories for debugging
    mpc_controller.lintraj.save(os.path.join(TARGET, 'mpc_reference.pkl'))
    campc_controller.lintraj.save(os.path.join(TARGET, 'campc_reference.pkl'))
    # Run the ambiguity optimization
    # ambi_model = campctools.run_ambiguity_optimization(campc_controller.getContactEstimationTrajectory())
    # campctools.compare_estimated_contact_model(ambi_model, true_plant.terrain, pts, savedir=TARGET, name='contactmodelwithambiguity')
    # # Save the contact model
    # utils.save(os.path.join(TARGET, 'contactambiguity.pkl'), ambi_model)

def main_ambiguity():
    plant = make_stepterrain_model()
    traj = campctools.load_estimation_trajectory(TARGET)
    model = campctools.run_ambiguity_optimization(traj)
    data = utils.load(os.path.join(TARGET, 'campcsim.pkl'))
    pts = campctools.get_x_samples(data, sampling=1000)
    campctools.compare_estimated_contact_model(model, plant.terrain, pts, savedir=TARGET, name='contactmodelwithambiguity')
    # Save the contact model
    utils.save(os.path.join(TARGET, 'contactambiguity.pkl'), model)

def main_animation():
    # Load the log files
    mpcfile = os.path.join(TARGET, 'mpc_logs', campctools.CONTROLLOGNAME)
    campcfile = os.path.join(TARGET, 'campc_logs', campctools.CONTROLLOGNAME)
    estfile = os.path.join(TARGET, 'campc_logs', campctools.ESTIMATELOGNAME)
    mpclogs = OptimizationLogger.load(mpcfile).logs
    campclogs = OptimizationLogger.load(campcfile).logs
    estlogs = OptimizationLogger.load(estfile).logs
    campc_full = [{**camplog, **estlog} for camplog, estlog in zip(campclogs, estlogs)]
    # Get the block model
    truemodel = make_stepterrain_model()
    # Load the appropriate reference trajectories
    reftraj = campctools.make_mpc_controller().lintraj
    esttraj = campctools.make_estimator_controller().getContactEstimationTrajectory()
    esttraj.loadEstimatedTrajectory(os.path.join(TARGET, campctools.TRAJNAME))
    # Setup the animator
    animation = animator.BlockCAMPCComparisonAnimator(truemodel, reftraj, esttraj)
    animation.animate(mpclogs, campc_full, savename=os.path.join(TARGET, ANIMATION_NAME))
    # mpc_animation = animator.BlockMPCAnimator(truemodel, reftraj)
    # mpc_animation.animate(mpclogs, savename=os.path.join(TARGET, MPCANIMATIONNAME))


if __name__ == '__main__':
    main_ambiguity()