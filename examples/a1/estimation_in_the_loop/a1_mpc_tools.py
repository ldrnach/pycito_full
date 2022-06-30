import os
import numpy as np
import matplotlib.pyplot as plt

import pycito.controller.mpc as mpc
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.systems.simulator import Simulator
import pycito.controller.mlcp as lcp


from pydrake.all import PiecewisePolynomial as pp

FIG_EXT = '.png'
CONTROLLOGFIG = 'mpclogs' + FIG_EXT
CONTROLLOGNAME = 'mpclogs.pkl'

LCP = lcp.ConstantRelaxedPseudoLinearComplementarityConstraint

def get_reference_trajectory(source):
    a1 = A1VirtualBase()
    a1.terrain.friction = 1.0
    a1.Finalize()
    return mpc.LinearizedContactTrajectory.loadLinearizedTrajectory(a1, source)

def set_controller_options(controller):
    a1 = controller.lintraj.plant
    Kp = np.ones((a1.multibody.num_positions(),))
    Kv = np.ones((a1.multibody.num_velocities(), ))
    Ks = np.diag(np.concatenate([1e2 * Kp, Kv], axis=0))
    controller.statecost = Ks
    controller.controlcost = 1e-4*np.eye(controller.control_dim)
    controller.forcecost = 1e-4 * np.eye(controller.force_dim)
    controller.slackcost = 1e-4 * np.eye(controller.slack_dim)
    controller.complementarity_schedule = [1e-2, 1e-4]    #originally 1e4
    controller.useSnoptSolver()
    controller.setSolverOptions({"Major feasibility tolerance": 1e-5,
                                "Major optimality tolerance": 1e-5,
                                'Scale option': 2,
                                'Major step limit':0.5,
                                'Superbasics limit':1000,
                                'Linesearch tolerance':0.1})
    controller.use_basis_file()
    controller.lintraj.useNearestTime()
    controller.enableLogging()
    return controller

def make_mpc_controller(reftraj, horizon=5):
    controller = mpc.LinearContactMPC(reftraj, horizon, lcptype=LCP)
    controller = set_controller_options(controller)
    return controller

def run_simulation(model, controller, duration=None):
    # Create the simulator
    simulator = Simulator(model, controller)
    simulator.useTimestepping()
    initial_state = controller.lintraj.getState(0)
    if duration is None:
        duration = controller.lintraj._time[-1]
    # Run the simulation
    tsim, xsim, usim, fsim, status = simulator.simulate(initial_state, duration)
    if not status:
        print(f"Simulation failed at timestep {tsim[-1]}")
    simdata = {'time': tsim, 
                'state': xsim,
                'control': usim,
                'force': fsim,
                'status': status}
    return simdata

def plot_mpc_logs(mpc_controller, savedir):
    print('Plotting MPC logs')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    mpc_controller.logger.plot(show=False, savename=os.path.join(savedir,CONTROLLOGFIG))

def save_mpc_logs(mpc_controller, savedir):
    print('Saving MPC logs')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    mpc_controller.logger.save(os.path.join(savedir, CONTROLLOGNAME))
    print('Saved!')

def plot_sim_results(plant, simdata, savedir=None, vis=False):
    """Plot the simulation trajectories"""
    if savedir is not None and not os.path.exists(savedir):
        os.makedirs(savedir)
    xtraj = pp.FirstOrderHold(simdata['time'], simdata['state'])
    utraj = pp.ZeroOrderHold(simdata['time'], simdata['control'])
    ftraj = pp.ZeroOrderHold(simdata['time'], simdata['force'])
    if savedir is None:
        plant.plot_trajectories(xtraj, utraj, ftraj, show=True)
    else:
        plant.plot_trajectories(xtraj, utraj, ftraj, show=False, savename=os.path.join(savedir, 'sim.png'))
    # Visualize in meshcat
    if vis:
        plant.visualize(xtraj)

def plot_tracking_error(simdata, reftraj, savedir):
    simstate = simdata['state']
    nT = min(simstate.shape[1], reftraj._state.shape[1]) - 1
    simstate = simstate[:, :nT]
    refstate = reftraj._state[:, :nT]
    if simdata['time'].shape[0] <= nT:
        time = simdata['time'][:nT]
    else:
        time = reftraj._time[:nT]
    # Calculate the errors
    nQ = int(simdata['state'].shape[0]/2)
    state_err = (refstate - simstate)**2
    base_err = np.sqrt(np.mean(state_err[:6, :], axis=0))
    joint_err = np.sqrt(np.mean(state_err[6:nQ,:],axis=0))
    bvel_err = np.sqrt(np.mean(state_err[nQ:nQ+6, :], axis=0))
    jvel_err = np.sqrt(np.mean(state_err[nQ+6:, :], axis=0))
    # Calculate the foot position errors
    a1 = A1VirtualBase()
    a1.Finalize()

    ref_feet = a1.state_to_foot_trajectory(refstate)
    sim_feet = a1.state_to_foot_trajectory(simstate)
    labels = ['FR','FL','BR','BL']
    feet = []
    for ref_foot, sim_foot in zip(ref_feet, sim_feet):
        err = np.mean((ref_foot - sim_foot)**2, axis=0)
        feet.append(np.sqrt(err))

    fig, axs = plt.subplots(3,1)
    # Configuration
    axs[0].plot(time, base_err, linewidth=1.5, label='base')
    axs[0].plot(time, joint_err, linewidth=1.5, label='joint')
    axs[0].set_ylabel('Position')
    axs[0].set_title('Root-Mean-Squared Tracking Error')
    # Velocity
    axs[1].plot(time, bvel_err, linewidth=1.5, label='base')
    axs[1].plot(time, jvel_err, linewidth=1.5, label='joint')
    axs[1].set_ylabel('Velocity')
    # Foot Position
    for foot, label in zip(feet, labels):
        axs[2].plot(time, foot, linewidth=1.5, label=label)
    axs[2].set_ylabel('Foot position')
    axs[2].set_xlabel('Time (s)')
    axs[0].legend(frameon=False, ncol=2)
    axs[2].legend(frameon=False, ncol=4)
    fig.savefig(os.path.join(savedir, 'trackingerror.png'), dpi=fig.dpi, bbox_inches='tight')