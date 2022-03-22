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


SAVEDIR = os.path.join('examples','sliding_block','mpc_speedtests')
FILENAME = 'speedtestresults.pkl'
FIGURENAME = 'SpeedTest.png'

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

def osqp_runner(test, target):
    test.useOsqpSolver()
    target = os.path.join(target, 'osqp')
    return test, target

def snopt_runner(test, target):
    test.useSnoptSolver()
    target = os.path.join(target, 'snopt')
    return test, target

def ipopt_runner(test, target):
    test.useIpoptSolver()
    target = os.path.join(target, 'ipopt')
    return test, target

def gurobi_runner(test, target):
    test.useGurobiSolver()
    target = os.path.join(target, 'gurobi')
    return test, target

def run_speedtests(runner = osqp_runner):
    block = create_block()
    # Load the reference trajectory
    file = os.path.join('data','slidingblock','block_trajopt.pkl')
    reftraj = mpc.LinearizedContactTrajectory.load(block, file)
    # Generate initial conditions
    N = 30
    x_initial = generate_initial_conditions(N)
    # Save the initial conditions
    # Create the target directory
    targetbase = SAVEDIR
    utils.save(os.path.join(targetbase, 'initialconditions.pkl'), x_initial)
    # Run the speedtests
    test = speedtesttools.MPCSpeedTest(reftraj)
    test, target = runner(test, targetbase)
    testResult = test.run_speedtests(x_initial)
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
    #run_speedtests(runner = ipopt_runner)
    run_speedtests(runner = gurobi_runner)
    #compare_speedtests()
    #replot_speedtest_results()