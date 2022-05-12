"""

"""
import numpy as np
import os
import matplotlib.pyplot as plt
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.systems.simulator import Simulator
import pycito.controller.mpc as mpc
import pycito.utilities as utils
from pycito.trajopt import complementarity as cp

from pydrake.all import PiecewisePolynomial as pp

SOURCE = os.path.join('data','a1','ellipse_foot_tracking','fast','fullstep','weight_1e+03','trajoptresults.pkl')
SAVEDIR = os.path.join('examples','a1','simulation_tests','fullstep')
FILENAME = 'simdata.pkl'
FIG_EXT = '.png'
CONTROLLOGFIG = 'mpclogs' + FIG_EXT
CONTROLLOGNAME = 'mpclogs.pkl'


def make_plant():
    a1 = A1VirtualBase()
    a1.terrain.friction = 1.0
    a1.Finalize()
    return a1

def make_plant_controller():
    a1 = make_plant()
    lintraj = mpc.LinearizedContactTrajectory.load(a1, utils.FindResource(SOURCE))
    controller = mpc.LinearContactMPC(lintraj, horizon=5)
    Kp = np.ones((a1.multibody.num_positions(),))
    Kv = np.ones((a1.multibody.num_velocities(), ))
    Ks = np.diag(np.concatenate([1e2 * Kp, Kv], axis=0))
    controller.statecost = Ks
    controller.controlcost = 1e-4*np.eye(controller.control_dim)
    controller.forcecost = 1e-4 * np.eye(controller.force_dim)
    controller.slackcost = 1e-4 * np.eye(controller.slack_dim)
    controller.complementaritycost = 1e4
    controller.useSnoptSolver()
    controller.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    controller.enableLogging()
    return controller

def run_simulation(simulator, initial_state, duration):
    tsim, xsim, usim, fsim, status = simulator.simulate(initial_state, duration)
    if not status:
        print(f"Simulation failed at timestep {tsim[-1]}")
    simdata = {'time': tsim, 
                'state': xsim,
                'control': usim,
                'force': fsim,
                'status': status}
    return simdata

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

def plot_tracking_error(sim, controller, savedir):
    nQ = controller.lintraj.plant.multibody.num_positions()
    nV = controller.lintraj.plant.multibody.num_velocities()
    t_sim = sim['time']
    x_sim = sim['state']
    t_idx = controller.lintraj.getTimeIndex(t_sim[-1]) + 1
    x_ref = controller.lintraj._state[:, :t_idx]
    #Base tracking error
    err = (x_sim - x_ref)**2
    base_err = np.sum(err[:6,:], axis=0)
    joint_err = np.sum(err[6:nQ,:], axis=0)
    bvel_err = np.sum(err[nQ:nQ+6, :], axis=0)
    jvel_err = np.sum(err[nQ+6:, :], axis=0)
    fig, axs = plt.subplots(2,1)
    axs[0].plot(t_sim, base_err, linewidth=1.5, label='Base')
    axs[0].plot(t_sim, joint_err, linewidth=1.5, label='Joint')
    axs[1].plot(t_sim, bvel_err, linewidth=1.5, label='Base')
    axs[1].plot(t_sim, jvel_err, linewidth=1.5, label='Joint')
    axs[0].set_ylabel('Position Error')
    axs[1].set_ylabel('Velocity Error')
    axs[1].set_xlabel('Time (s)')
    axs[0].set_title('Tracking Error')
    axs[0].legend(frameon = False)
    fig.savefig(os.path.join(savedir, 'trackingerror.png'), dpi=fig.dpi, bbox_inches='tight')

def run_timestepping():
    filepart = 'timestepping'
    print(f"Running simulation with {filepart} integrator")
    a1 = make_plant()
    controller = make_plant_controller()
    simulator = Simulator(a1, controller)
    simulator.useTimestepping()
    x0 = controller.lintraj.getState(0)
    duration = controller.lintraj._time[-1]
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(a1, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)
    plot_mpc_logs(controller, os.path.join(SAVEDIR, filepart,'mpclogs'))
    save_mpc_logs(controller, os.path.join(SAVEDIR, filepart,'mpclogs'))
    plot_tracking_error(simresults, controller, os.path.join(SAVEDIR, filepart))

def run_implicit():
    filepart = 'implicit'
    print(f"Running simulation with {filepart} integrator")
    a1 = make_plant()
    controller = make_plant_controller()
    simulator = Simulator(a1, controller)
    simulator.useImplicitEuler(ncp=cp.NonlinearVariableSlackComplementarity)
    x0 = controller.lintraj.getState(0)
    duration = controller.lintraj._time[-1]
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(a1, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)
    plot_mpc_logs(controller, os.path.join(SAVEDIR, filepart,'mpclogs'))
    save_mpc_logs(controller, os.path.join(SAVEDIR, filepart,'mpclogs'))


def run_semiimplicit():
    filepart = 'semiimplicit'
    print(f"Running simulation with {filepart} integrator")
    a1 = make_plant()
    controller = make_plant_controller()
    simulator = Simulator(a1, controller)
    simulator.useSemiImplicitEuler(ncp=cp.NonlinearVariableSlackComplementarity)
    x0 = controller.lintraj.getState(0)
    duration = controller.lintraj._time[-1]
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(a1, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)
    plot_mpc_logs(controller, os.path.join(SAVEDIR, filepart,'mpclogs'))
    save_mpc_logs(controller, os.path.join(SAVEDIR, filepart,'mpclogs'))


def run_midpoint():
    filepart = 'midpoint'
    print(f"Running simulation with {filepart} integrator")
    a1 = make_plant()
    controller = make_plant_controller()
    simulator = Simulator(a1, controller)
    simulator.useImplicitMidpoint(ncp=cp.NonlinearVariableSlackComplementarity)
    x0 = controller.lintraj.getState(0)
    duration = controller.lintraj._time[-1]
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(a1, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)
    plot_mpc_logs(controller, os.path.join(SAVEDIR, filepart,'mpclogs'))
    save_mpc_logs(controller, os.path.join(SAVEDIR, filepart,'mpclogs'))


if __name__ == '__main__':
    run_timestepping()
    #run_implicit()
    #run_semiimplicit()
    #run_midpoint()