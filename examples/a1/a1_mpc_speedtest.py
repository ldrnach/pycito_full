"""
Speedtests for A1

Luke Drnach
February 22, 2022
"""

import numpy as np
import os

from pycito.systems.A1.a1 import A1VirtualBase
import pycito.controller.mpc as mpc
import pycito.controller.speedtesttools as speedtesttools
import pycito.utilities as utils

def create_a1():
    plant = A1VirtualBase()
    plant.Finalize()
    return plant

def generate_initial_conditions(x0, N = 30):
    # First draw a sequence of initial states
    noises = np.random.default_rng(seed=42).standard_normal((2, N))
    # Now clip the z-axis values to ensure the states are feasible
    noises[1, noises[1,:] < 0] = 0
    states = np.zeros((x0.shape[0], N))
    for k, noise in enumerate(noises.T):
        states[:, k] = x0.copy()
        states[[0, 2, ], k] += noise  
    return states

def run_speedtests():
    plant = create_a1()
    # Load the reference trajectory
    file = os.path.join('examples','a1','foot_tracking_gait','')
    reftraj = mpc.LinearizedContactTrajectory.load(plant, file)
    # Generate initial conditions
    N = 30
    x0 = reftraj.getState(0)
    x_initial = generate_initial_conditions(x0, N)
    # Save the initial conditions
    # Create the target directory
    targetbase = os.path.join('examples','a1','mpc_speedtests')
    utils.save(os.path.join(targetbase, 'initialconditions.pkl'), x_initial)
    # Run the OSQP speedtests
    test = speedtesttools.MPCSpeedTest(reftraj)
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
