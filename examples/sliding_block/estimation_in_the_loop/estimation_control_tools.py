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
#TODO: Solve ERROR 41 in MPC
#TODO: Solve problems with friction updating from terrain.py
#TODO: FIX: Semiparametric models return different shapes for prior and posterior evaluations

import os, copy
import numpy as np
import matplotlib.pyplot as plt
import pycito.systems.contactmodel as cm
import pycito.controller.mpc as mpc
import pycito.controller.mlcp as mlcp
import pycito.controller.contactestimator as ce
from pycito.systems.simulator import Simulator
import pycito.utilities as utils
from pycito.systems.block.block import Block

FIG_EXT = '.png'
MPC_HORIZON = 5
ESTIMATION_HORIZON = 5
REFSOURCE = os.path.join('data','slidingblock','block_reference.pkl')
TARGET = os.path.join('examples','sliding_block','estimation_control')
TERRAINFIGURE = 'EstimatedTerrain' + FIG_EXT
TRAJNAME = 'estimatedtrajectory.pkl'
CONTROLLOGFIG = 'mpclogs' + FIG_EXT
CONTROLLOGNAME = 'mpclogs.pkl'
ESTIMATELOGFIG = 'EstimationLogs' + FIG_EXT
ESTIMATELOGNAME = 'EstimationLogs.pkl'

def make_semiparametric_block_model():
    block = Block()
    block.Finalize()
    block.terrain = cm.SemiparametricContactModel.FlatSurfaceWithRBFKernel(friction = 0.5,
                                                                         length_scale = np.array([0.1, 0.1, np.inf]),    
                                                                         reg=0.01)
    return block 

def make_block_model():
    block = Block()
    block.Finalize()
    return block

def make_estimator_controller():
    block = make_semiparametric_block_model()
    reftraj = mpc.LinearizedContactTrajectory.load(block, REFSOURCE)
    # Create the estimator
    x0 = reftraj.getState(0)
    block2 = make_semiparametric_block_model()
    esttraj = ce.ContactEstimationTrajectory(block2, x0)
    estimator = ce.ContactModelEstimator(esttraj, ESTIMATION_HORIZON)
    # Set the estimator solver parameters
    estimator.forcecost = 1e-1
    estimator.relaxedcost = 1e4
    estimator.distancecost = 1e-3
    estimator.frictioncost = 1e-3
    estimator.useSnoptSolver()
    estimator.setSolverOptions({"Major feasibility tolerance": 1e-4,
                                "Major optimality tolerance": 1e-4})
    # Create the overall controller
    controller = mpc.ContactAdaptiveMPC(estimator, reftraj, MPC_HORIZON)
    # Tune the controller
    controller = set_controller_options(controller)

    return controller

def set_controller_options(controller):
    controller.statecost = np.diag([1e2, 1e2, 1, 1])
    controller.controlcost = 1e-2 * np.eye(controller.control_dim)
    controller.forcecost = 1e-4 * np.eye(controller.force_dim)
    controller.slackcost = 1e-2 * np.eye(controller.slack_dim)
    controller.complementaritycost = 1e3
    controller.useSnoptSolver()
    controller.setSolverOptions({"Major feasibility tolerance": 1e-4,
                                "Major optimality tolerance": 1e-4,
                                'Scale option': 1})
    controller.use_random_guess()
    controller.lintraj.useNearestPosition()
    return controller

def make_mpc_controller():
    block = make_block_model()
    reftraj = mpc.LinearizedContactTrajectory.load(block, REFSOURCE)
    controller = mpc.LinearContactMPC(reftraj, MPC_HORIZON)
    controller = set_controller_options(controller)
    return controller

def run_simulation(plant, controller, initial_state, duration, savedir=None):
    #Check if the target directory exists
    if savedir is not None and not os.path.exists(savedir):
        os.makedirs(savedir)
    # Create and run the simulation
    sim = Simulator(plant, controller)
    sim.useTimestepping()
    tsim, xsim, usim, fsim, status = sim.simulate(initial_state, duration)
    if not status:
        print(f"Simulation faied at timestep {tsim[-1]}")
    sim_results = {'time': tsim,
                    'state': xsim,
                    'control': usim,
                    'force': fsim,
                    'status': status}
    return sim_results

def plot_mpc_logs(mpc_controller, savedir):
    print('Plotting MPC logs')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    mpc_controller.logger.plot(show=False, savename=os.path.join(savedir,CONTROLLOGFIG))

def plot_campc_logs(campc_controller, savedir):
    print('Plotting CAMPC logs')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    campc_controller.getControllerLogs().plot(show=False, savename=os.path.join(savedir, CONTROLLOGFIG))
    campc_controller.getEstimatorLogs().plot(show=False, savename=os.path.join(savedir, ESTIMATELOGFIG))

def save_mpc_logs(mpc_controller, savedir):
    print('Saving MPC logs')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    mpc_controller.logger.save(os.path.join(savedir, CONTROLLOGNAME))
    print('Saved!')

def save_campc_logs(campc_controller, savedir):
    print('Saving CAMPC logs')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    campc_controller.getControllerLogs().save(os.path.join(savedir, CONTROLLOGNAME))
    campc_controller.getEstimatorLogs().save(os.path.join(savedir, ESTIMATELOGNAME))
    print('Saved!')

def plot_terrain_errors(controller, savedir):
    print('Plotting terrain residuals')
    esttraj = controller.getContactEstimationTrajectory()
    plotter = ce.ContactEstimationPlotter(esttraj)
    plotter.plot(show=False, savename=os.path.join(savedir, TERRAINFIGURE))

def save_estimated_terrain(controller, savedir):
    print("saving estimated terrain")
    controller.getContactEstimationTrajectory().save(os.path.join(savedir, TRAJNAME))
    print("Saved!")

def plot_trajectory_comparison(mpc_sim, campc_sim, savename):
    print('Generating simulation comparison plots')
    fig, axs = plt.subplots(3,1)
    # Horizontal state plot
    plot_horizontal_sim_trajectory(axs, mpc_sim, label='MPC')
    plot_horizontal_sim_trajectory(axs, campc_sim, label='CAMPC')
    axs[0].set_title('Horizontal Trajectory')
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(savename, 'horizontal_traj' + FIG_EXT), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)
    # Vertical State Plot
    fig, axs = plt.subplots(2,1)
    plot_vertical_sim_trajectory(axs, mpc_sim, label='MPC')
    plot_vertical_sim_trajectory(axs, campc_sim, label='CAMPC')
    axs[0].set_title('Vertical Trajectory')
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(savename, 'vertical_traj' + FIG_EXT), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)
    # Ground reaction force plot
    fig, axs = plt.subplots(3,1)
    plot_sim_forces(axs, mpc_sim, label='MPC')
    plot_sim_forces(axs, campc_sim, label='CAMPC')
    axs[0].set_title('Reaction Forces')
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(savename, 'reaction_forces' + FIG_EXT), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)

def plot_horizontal_sim_trajectory(axs, sim, label=None):
    axs[0].plot(sim['time'], sim['state'][0,:], linewidth=1.5, label=label)
    axs[0].set_ylabel('Position (m)')
    axs[1].plot(sim['time'], sim['state'][2,:], linewidth=1.5, label=label)
    axs[1].set_ylabel('Velocity (m/s)')
    axs[2].plot(sim['time'], sim['control'][0,:], linewidth=1.5, label=label)
    axs[2].set_ylabel('Control (N)')
    axs[2].set_xlabel('Time (s)')

def plot_vertical_sim_trajectory(axs, sim, label=None):
    axs[0].plot(sim['time'], sim['state'][1,:], linewidth=1.5, label=label)
    axs[0].set_ylabel('Position (m)')
    axs[1].plot(sim['time'], sim['state'][3,:], linewidth=1.5, label=label)
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_xlabel('Time (s)')

def plot_sim_forces(axs, sim, label=None):
    axs[0].plot(sim['time'], sim['force'][0,:], linewidth=1.5, label=label)
    axs[0].set_ylabel('Normal (N)')
    axs[1].plot(sim['time'], sim['force'][1,:] - sim['force'][3,:], linewidth=1.5, label=label)
    axs[1].set_ylabel('Friction-X (N)')
    axs[2].plot(sim['time'], sim['force'][2,:] - sim['force'][4,:], linewidth=1.5, label=label)
    axs[2].set_ylabel('Friction-Y (N)')
    axs[2].set_xlabel('Time (s)')

def compare_forces(sim, campc, savedir):
    fig, axs = plt.subplots(3,1)
    plot_sim_forces(axs, sim, label='True')
    campc_data = {'time': np.asarray(campc.getContactEstimationTrajectory()._time),
                'force': np.column_stack(campc.getContactEstimationTrajectory()._forces)
    }
    plot_sim_forces(axs, campc_data, label='Estimated')
    axs[0].set_title('Estimated Force Comparison')
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(savedir, 'estimated_force_comparison' + FIG_EXT), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)

def get_contact_model(campc):
    etraj = campc.getContactEstimationTrajectory()
    model = copy.deepcopy(etraj.contact_model)
    N = etraj.num_timesteps
    cpts = np.concatenate(etraj.get_contacts(0, N), axis=1)
    derr = np.row_stack(etraj.get_distance_error(0, N))
    ferr = np.row_stack(etraj.get_friction_error(0, N))
    Kd, Kf = etraj.getContactKernels(0, N)
    dweight = np.linalg.lstsq(Kd, derr, rcond=None)[0]
    fweight = np.linalg.lstsq(Kf, ferr, rcond=None)[0]
    model.add_samples(cpts, dweight, fweight)
    return model

def compare_estimated_contact_model(estimated, true, pts, savedir, name='estimatedcontactmodel'):
    print(f"Generating contact model comparison plots")
    fig, axs = true.plot2D(pts, label='True', show=False, savename=None)
    fig, axs = estimated.plot2D(pts, axs, label='Estimated', show=False, savename=None)
    axs[0].set_title('Contact Model Estimation Performance')
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(savedir, name + FIG_EXT), dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)

def get_x_samples(sim, sampling=100):
    xvals = sim['state'][0,:]
    pt0 = np.zeros((3,))
    ptN = np.zeros((3,))
    pt0[0], ptN[0] = np.amin(xvals), np.amax(xvals)
    return np.linspace(pt0, ptN, sampling).transpose()

def run_ambiguity_optimization(esttraj):
    rectifier = ce.EstimatedContactModelRectifier(esttraj, surf_max = 100, fric_max = 100)
    rectifier.useSnoptSolver()
    rectifier.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    ambi_model = rectifier.solve_global_model_with_ambiguity()
    return ambi_model

def load_estimation_trajectory(loaddir):
    block = make_semiparametric_block_model()
    return ce.ContactEstimationTrajectory.load(block, os.path.join(loaddir, 'estimatedtrajectory.pkl'))

if __name__ == '__main__':
    print("Heelo from estimation_control_tools.py!")