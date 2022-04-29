"""

"""
import numpy as np
import os
from pycito.systems.block.block import Block
from pycito.systems.simulator import Simulator
import pycito.controller.mpc as mpc
import pycito.utilities as utils

from pydrake.all import PiecewisePolynomial as pp

SOURCE = os.path.join('data','slidingblock','block_reference.pkl')
SAVEDIR = os.path.join('examples','sliding_block','simulation_tests')
FILENAME = 'simdata.pkl'

def make_block():
    block = Block()
    block.Finalize()
    return block

def make_block_controller():
    block = make_block()
    data = utils.load(utils.FindResource(SOURCE))
    return mpc.OpenLoopController(block, data['time'], data['control'])


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
    """Plot the block trajectories"""
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
    block = make_block()
    controller = make_block_controller()
    simulator = Simulator(block, controller)
    simulator.useTimestepping()
    x0 = np.array([0, 0.5, 0, 0])
    duration = 1.5
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(block, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)

def run_implicit():
    filepart = 'implicit'
    print(f"Running simulation with {filepart} integrator")
    block = make_block()
    controller = make_block_controller()
    simulator = Simulator(block, controller)
    simulator.useImplicitEuler()
    x0 = np.array([0, 0.5, 0, 0])
    duration = 1.5
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(block, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)

def run_semiimplicit():
    filepart = 'semiimplicit'
    print(f"Running simulation with {filepart} integrator")
    block = make_block()
    controller = make_block_controller()
    simulator = Simulator(block, controller)
    simulator.useSemiImplicitEuler()
    x0 = np.array([0, 0.5, 0, 0])
    duration = 1.5
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(block, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)

def run_midpoint():
    filepart = 'midpoint'
    print(f"Running simulation with {filepart} integrator")
    block = make_block()
    controller = make_block_controller()
    simulator = Simulator(block, controller)
    simulator.useImplicitMidpoint()
    x0 = np.array([0, 0.5, 0, 0])
    duration = 1.5
    simresults = run_simulation(simulator, x0, duration)
    plot_sim_results(block, simresults, savedir=os.path.join(SAVEDIR, filepart))
    utils.save(os.path.join(SAVEDIR, filepart, FILENAME), simresults)

if __name__ == '__main__':
    run_timestepping()
    run_implicit()
    run_semiimplicit()
    run_midpoint()