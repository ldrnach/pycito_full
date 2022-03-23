"""
Speedtests for the sliding block example

Luke Drnach
March 9, 2022
"""
import numpy as np
import os

from pycito.systems.block.block import Block
from pycito.systems.contactmodel import SemiparametricContactModel
import pycito.controller.contactestimator as ce
import pycito.controller.speedtesttools as speedtesttools

import pycito.utilities as utils

SAVEDIR = os.path.join('examples','sliding_block','estimator_speedtests_debug')
FILENAME = 'speedtestresults.pkl'
FIGURENAME = 'SpeedTest.png'

SOURCEDATA = os.path.join('openloop','simdata.pkl')
SOURCEDIR = os.path.join('examples','sliding_block','simulations')

def create_block():
    block = Block()
    block.Finalize()
    block.terrain = SemiparametricContactModel.FlatSurfaceWithRBFKernel()
    return block

def datakeys():
    return ['state', 'control', 'force']

def get_data(sourcepart):
    """Load the data defined by the subdirectory 'sourcepart' """
    # Load the data
    file = os.path.join(SOURCEDIR, sourcepart, SOURCEDATA)
    data = utils.load(utils.FindResource(file))
    # Truncate the data after 110 points
    data['time'] = data['time'][:111]
    for key in datakeys():
        data[key] = data[key][:, :111]
    return data

def replot_speedtest_results(source):
    filename = os.path.join(SAVEDIR, source, FILENAME)
    result = speedtesttools.SpeedTestResult.load(filename)
    target = os.path.join(SAVEDIR, source, FIGURENAME)
    result.plot(show=False, savename=target)

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

def run_speedtests(source, method = snopt_runner):
    """Main script for running estimation speedtests"""
    print(f"Running contact estimation speedtests for {source}")
    # Create the estimation trajectory from the source data
    data = get_data(source)
    block = create_block()
    traj = ce.ContactEstimationTrajectory(block, data['state'][:, 0])
    for t, x, u in zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 1:].T):
        traj.append_sample(t, x, u)
    # Now setup the estimation speedtest
    numstarts = 30
    maxhorizon = 50
    # Make the speedtest
    speedtest = speedtesttools.ContactEstimatorSpeedTest(traj, maxhorizon)
    speedtest, target = method(speedtest, SAVEDIR)
    result = speedtest.run_speedtests(numstarts)
    # Save the results
    target = os.path.join(target, source)
    if not os.path.exists(target):
        os.makedirs(target)
    result.save(os.path.join(target, FILENAME))
    result.plot(show=False, savename=os.path.join(target, FIGURENAME))

def flat_terrain_speedtests(method = snopt_runner):
    """Run estimation speedtests on flat terrain data"""
    source = 'flatterrain'
    run_speedtests(source, method)

def step_terrain_speedtests(method = snopt_runner):
    """Run estimation speedtests on stepped terrain data"""
    source = 'steppedterrain'
    run_speedtests(source, method)

def low_friction_speedtests(method = snopt_runner):
    """Run estimation speedtests on terrain with a step change in friction"""
    source='lowfriction'
    run_speedtests(source, method)

if __name__ == '__main__':
    #replot_speedtest_results('flatterrain')
    flat_terrain_speedtests()
    step_terrain_speedtests()
    low_friction_speedtests()
