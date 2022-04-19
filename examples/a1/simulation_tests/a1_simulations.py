"""

"""
import numpy as np
import os
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.systems.simulator import Simulator
import pycito.controller.mpc as mpc
import pycito.utilities as utils
from pycito.trajopt import complementarity as cp

from pydrake.all import PiecewisePolynomial as pp

SOURCE = os.path.join('data','a1','a1_step.pkl')
SAVEDIR = os.path.join('examples','a1','simulation_tests')
FILENAME = 'simdata.pkl'

def make_plant():
    a1 = A1VirtualBase()
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
    controller.controlcost = 1e-2*np.eye(controller.control_dim)
    controller.forcecost = 1e-2 * np.eye(controller.force_dim)
    controller.slackcost = 1e-2 * np.eye(controller.slack_dim)
    controller.complementaritycost = 1e2
    controller.useSnoptSolver()
    controller.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
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

def run_implicit():
    filepart = 'implicit'
    print(f"Running simulation with {filepart} integrator")
    a1 = make_plant()
    controller = make_plant_controller()
    simulator = Simulator(a1, controller)
    simulator.useImplicitEuler(ncp=cp.CostRelaxedLinearEqualityComplementarity)
    x0 = controller.lintraj.getState(0)
    duration = controller.lintraj._time[-1]
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(a1, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)

def run_semiimplicit():
    filepart = 'semiimplicit'
    print(f"Running simulation with {filepart} integrator")
    a1 = make_plant()
    controller = make_plant_controller()
    simulator = Simulator(a1, controller)
    simulator.useSemiImplicitEuler(ncp=cp.CostRelaxedLinearEqualityComplementarity)
    x0 = controller.lintraj.getState(0)
    duration = controller.lintraj._time[-1]
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(a1, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)

def run_midpoint():
    filepart = 'midpoint'
    print(f"Running simulation with {filepart} integrator")
    a1 = make_plant()
    controller = make_plant_controller()
    simulator = Simulator(a1, controller)
    simulator.useImplicitMidpoint(ncp=cp.CostRelaxedLinearEqualityComplementarity)
    x0 = controller.lintraj.getState(0)
    duration = controller.lintraj._time[-1]
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(a1, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)

if __name__ == '__main__':
    #run_timestepping()
    run_implicit()
    #run_semiimplicit()
    #run_midpoint()