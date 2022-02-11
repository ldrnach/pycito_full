"""
Speedtests for the sliding block example

Luke Drnach
Feburary 22, 2022
"""

import numpy as np
import os

from pycito.systems.block.block import Block
import pycito.controller.mpc as mpc
import pycito.controller.speedtesttools as speedtesttools
import pycito.utilities as utils

def create_block():
    block = Block()
    block.Finalize()
    return block

def generate_initial_conditions(N = 30):
    # First draw a sequence of initial states
    x_initial = np.random.default_rng(seed=42).standard_normal((4, N))
    x_initial[1,:] += 0.5
    # Now clip the z-axis values to ensure the states are feasible
    x_initial[1, x_initial[1,:] < 0] = 0
    return x_initial

def run_speedtests():
    block = create_block()
    # Load the reference trajectory
    file = os.path.join('data','slidingblock','block_trajopt.pkl')
    reftraj = mpc.LinearizedContactTrajectory.load(block, file)
    # Generate initial conditions
    N = 30
    x_initial = generate_initial_conditions(N)
    # Save the initial conditions
    # Create the target directory
    targetbase = os.path.join('examples','sliding_block','mpc_speedtests')
    utils.save(os.path.join(targetbase, 'initialconditions.pkl'), x_initial)
    # Run the OSQP speedtests
    test = speedtesttools.MPCSpeedTest(reftraj, max_horizon=2)
    test.useOsqpSolver()
    testResult = test.run_speedtests(x_initial)
    target = os.path.join(targetbase, 'osqp')
    if not os.path.exists(target):
        os.makedirs(target)
    testResult.save(os.path.join(target, 'speedtestresults.pkl'))
    testResult.plot(show=False, savename=os.path.join(target, 'SpeedTest.png'))
    # Run the SNOPT speedtests
    test.useSnoptSolver()
    testResult = test.run_speedtests(x_initial)
    target = os.path.join(targetbase, 'snopt')
    if not os.path.exists(target):
        os.makedirs(target)
    testResult.save(os.path.join(target, 'speedtestresults.pkl'))
    testResult.plot(show=False, savename=os.path.join(target, 'SpeedTest.png'))

if __name__ == '__main__':
    run_speedtests()
