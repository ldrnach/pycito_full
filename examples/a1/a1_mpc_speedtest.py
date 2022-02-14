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

SAVEDIR = os.path.join('examples', 'a1','mpc_speedtests')
FILENAME = 'speedtestresults.pkl'
FIGURENAME = 'SpeedTest.png'

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
    file = os.path.join('examples','a1','foot_tracking_gait','first_step','weight_1e+03','trajoptresults.pkl')
    reftraj = mpc.LinearizedContactTrajectory.load(plant, file)
    # Generate initial conditions
    N = 30
    x0 = reftraj.getState(0)
    x_initial = generate_initial_conditions(x0, N)
    # Save the initial conditions
    # Create the target directory
    targetbase = SAVEDIR
    utils.save(os.path.join(targetbase, 'initialconditions.pkl'), x_initial)
    # Run the OSQP speedtests
    test = speedtesttools.MPCSpeedTest(reftraj)
    test.useOsqpSolver()
    testResult = test.run_speedtests(x_initial)
    target = os.path.join(targetbase, 'osqp')
    if not os.path.exists(target):
        os.makedirs(target)
    testResult.save(os.path.join(target, FILENAME))
    testResult.plot(show=False, savename=os.path.join(target, FIGURENAME))
    # Run the SNOPT speedtests
    test.useSnoptSolver()
    testResult = test.run_speedtests(x_initial)
    target = os.path.join(targetbase, 'snopt')
    if not os.path.exists(target):
        os.makedirs(target)
    testResult.save(os.path.join(target, FILENAME))
    testResult.plot(show=False, savename=os.path.join(target, FIGURENAME))

def replot_speedtest_results():
    """Helper function for perfecting speedtest result plots without re-running speedtests"""
    basedir = SAVEDIR
    sources = ['osqp', 'snopt']
    for source in sources:
        figure = os.path.join(basedir, source, FIGURENAME)
        results = speedtesttools.SpeedTestResult.load(os.path.join(basedir, source, FILENAME))
        results.plot(show=False, savename=figure)
        print(f"Saved new figures to {figure}")

def compare_speedtests():
    """Helper function for plotting the solver speedtest results"""
    sources = ['osqp','snopt']
    results = [speedtesttools.SpeedTestResult.load(os.path.join(SAVEDIR, source, FILENAME)) for source in sources]
    savename = os.path.join(SAVEDIR,'speedtestcomparison.png')
    speedtesttools.SpeedTestResult.compare_results(results, sources, show=False, savename=savename)
    print(f"Saved comparison figures to {SAVEDIR}")

if __name__ == '__main__':
    compare_speedtests()
