"""
Comparison of open and closed loop simulations using the sliding block

Luke Drnach
February 8, 2022
"""
import os

from pydrake.all import PiecewisePolynomial as pp

from pycito.systems.block.block import Block
from pycito.systems.simulator import Simulator
import pycito.controller.mpc as mpc
import pycito.utilities as utils

#TODO: Check on control signal. The final value is nonzero, which causes the block to slide backwards after 1s. 

def plot_sim_results(plant, t, x, u, f, savedir, vis=True):
    """Plot the block trajectories"""
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    xtraj = pp.FirstOrderHold(t, x)
    utraj = pp.ZeroOrderHold(t, u)
    ftraj = pp.ZeroOrderHold(t, f)
    plant.plot_trajectories(xtraj, utraj, ftraj, show=False, savename=os.path.join(savedir, 'sim.png'))
    # Visualize in meshcat
    if vis:
        plant.visualize(xtraj)


def main():
    # Data source
    source = os.path.join('data','slidingblock','block_trajopt.pkl')
    savedir = os.path.join('examples','sliding_block','simulations')
    # Make the plant model
    block = Block()
    block.Finalize()
    # Create the controller
    reftraj = mpc.LinearizedContactTrajectory.load(block, source)
    controller = mpc.LinearContactMPC(reftraj, horizon=10)
    # Get the open loop control
    utraj = pp.ZeroOrderHold(reftraj._time, reftraj._control)
    # Setup solver options
    controller.useSnoptSolver()
    controller.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    # Create the simulator
    opensim = Simulator.OpenLoop(block, utraj)
    closedsim = Simulator.ClosedLoop(block, controller)
    # Run the simulations
    initial_state = reftraj.getState(0)
    print(f"Running open loop simulation")
    topen, xopen, uopen, fopen = opensim.simulate(initial_state, duration=1.0)
    print(f"Running closed loop simulation")
    tcl, xcl, ucl, fcl = closedsim.simulate(initial_state, duration=1.0)
    # Plot the results
    plot_sim_results(block, topen, xopen, uopen, fopen, savedir=os.path.join(savedir, 'open_loop'))
    plot_sim_results(block, tcl, xcl, ucl, fcl, savedir=os.path.join(savedir, 'closed_loop'))
    # Save the data
    opendata = {'time': topen,
                'state': xopen,
                'control': uopen,
                'force': fopen}
    utils.save(os.path.join(savedir, 'open_loop','simdata.pkl'), opendata)
    closeddata = {'time': tcl,
                'state': xcl,
                'control': ucl,
                'force': fcl}
    utils.save(os.path.join(savedir, 'closed_loop','simdata.pkl'), closeddata)
    
if __name__ == '__main__':

    main()


