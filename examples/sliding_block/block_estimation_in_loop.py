"""
block_estimation_in_loop:
    Example script demonstrating contact estimation in the control loop for the sliding block example. Examples inlcude:
        1. Flatterrain - where there is no model mismatch between the planning and execution model
        2. Highfriction - where the execution model has higher friction than the planning model
        3. Lowfriction - where the execution model has lower friction than the planning model
        4. Steppedterrain - where the execution model has a sudden drop in the terrain height

Luke Drnach
April 11, 2022
"""
import os
import numpy as np

import examples.sliding_block.blocktools as blocktools
import pycito.controller.mpc as mpc
import pycito.controller.contactestimator as ce
from pycito.systems.simulator import Simulator
import pycito.utilities as utils

MPC_HORIZON = 5
ESTIMATION_HORIZON = 5
REFSOURCE = os.path.join('data','slidingblock','block_trajopt.pkl')
TARGET = os.path.join('examples','sliding_block','estimation_control')
TERRAINFIGURE = 'EstimatedTerrain.png'
TRAJNAME = 'estimatedtrajectory.pkl'
CONTROLLOGFIG = 'MPCLogs.png'
CONTROLLOGNAME = 'mpclogs.pkl'
ESTIMATELOGFIG = 'EstimationLogs.png'
ESTIMATELOGNAME = 'EstimationLogs.pkl'

def make_estimator_controller():
    block = blocktools.make_semiparametric_block_model()
    reftraj = mpc.LinearizedContactTrajectory.load(block, REFSOURCE)
    # Create the estimator
    t0 = reftraj.getTime(0)
    x0 = reftraj.getState(0)
    esttraj = ce.ContactEstimationTrajectory(block, x0)
    estimator = ce.ContactModelEstimator(esttraj, ESTIMATION_HORIZON)
    # Set the estimator solver parameters
    estimator.forcecost = 1e-1
    estimator.relaxedcost = 1e4
    estimator.distancecost = 1e-3
    estimator.frictioncost = 1e-3
    estimator.useSnoptSolver()
    estimator.setSolverOptions({"Major feasibility tolerance": 1e-6,
                                "Major optimality tolerance": 1e-6})
    # Create the overall controller
    controller = mpc.ContactAdaptiveMPC(estimator, reftraj, MPC_HORIZON)
    # Tune the controller
    controller.statecost = 1e2 * np.eye(controller.state_dim)
    controller.controlcost = 1e-2 * np.eye(controller.control_dim)
    controller.forcecost = 1e-2 * np.eye(controller.force_dim)
    controller.slackcost = 1e-2 * np.eye(controller.slack_dim)
    controller.useSnoptSolver()
    controller.setSolverOptions({"Major feasibility tolerance": 1e-6,
                                "Major optimality tolerance": 1e-6})
    # Enable logging
    controller.enableLogging()
    return controller

def run_simulation(plant, controller, initial_state, duration, savedir=None):
    #Check if the target directory exists
    if savedir is not None and not os.path.exists(savedir):
        os.makedirs(savedir)
    # Create and run the simulation
    sim = Simulator(plant, controller)
    tsim, xsim, usim, fsim, status = sim.simulate(initial_state, duration)
    if not status:
        print(f"Simulation faied at timestep {tsim[-1]}")
    blocktools.plot_sim_results(plant, tsim, xsim, usim, fsim, savedir)
    # Save the sim data
    blocktools.save_sim_data(tsim, xsim, usim, fsim, savedir)
    # Plot the estimated terrain
    plot_estimated_terrain(controller, savedir)
    save_estimation_control_logs(controller, savedir)

def plot_estimated_terrain(controller, savedir):
    esttraj = controller.getContactEstimationTrajectory()
    plotter = ce.ContactEstimationPlotter(esttraj)
    plotter.plot(show=False, savename=os.path.join(savedir, TERRAINFIGURE))
    esttraj.saveContactTrajectory(os.path.join(savedir, TRAJNAME))

def save_estimation_control_logs(controller, savedir):
    controllogs = controller.getControllerLogs()
    controllogs.plot(savename=os.path.join(savedir, CONTROLLOGFIG))
    controllogs.save(os.path.join(savedir, CONTROLLOGNAME))
    estimatelogs = controller.getEstimatorLogs()
    estimatelogs.plot(savename=os.path.join(savedir, ESTIMATELOGFIG))
    estimatelogs.save(os.path.join(savedir, ESTIMATELOGNAME))

def main_flatterrain():
    block = blocktools.make_flatterrain_model()
    controller = make_estimator_controller()
    x0 = controller.lintraj.getState(0)
    target = os.path.join(TARGET, 'flatterrain')
    run_simulation(block, controller, x0, duration=1.5, savedir=target)

def main_lowfriction():
    block = blocktools.make_lowfriction_model()
    controller = make_estimator_controller()
    x0 = controller.lintraj.getState(0)
    target = os.path.join(TARGET, 'lowfriction')
    run_simulation(block, controller, x0, duration=1.5, savedir=target) 

def main_highfriction():
    block = blocktools.make_highfriction_model()
    controller = make_estimator_controller()
    x0 = controller.lintraj.getState(0)
    target = os.path.join(TARGET, 'highfriction')
    run_simulation(block, controller, x0, duration=1.5, savedir=target) 

def main_stepterrain():
    block = blocktools.make_stepterrain_model()
    controller = make_estimator_controller()
    x0 = controller.lintraj.getState(0)
    target = os.path.join(TARGET, 'stepterrain')
    run_simulation(block, controller, x0, duration=1.5, savedir=target) 

if __name__ == '__main__':
    main_flatterrain()