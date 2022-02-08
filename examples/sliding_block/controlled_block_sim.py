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

def main():
    # Data source
    source = os.path.join('data','slidingblock','block_trajopt.pkl')
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
    


if __name__ == '__main__':
    main()


