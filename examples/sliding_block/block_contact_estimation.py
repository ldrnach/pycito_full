import os
import numpy as np
import matplotlib.pyplot as plt
from pycito.controller import contactestimator as ce
from pycito.systems.block.block import Block
from pycito.systems.contactmodel import SemiparametricContactModel
import pycito.utilities as utils
from pycito.controller.optimization import OptimizationLogger

SOURCEBASE = os.path.join('examples','sliding_block','simulations','correct')
SOURCEFILE = os.path.join('mpc', 'simdata.pkl')
SAVEDIR = os.path.join('examples','sliding_block','estimation_mpc','SNR1e5_N1_Feasibility')
FIGURENAME = 'EstimationResults.png'
LOGFIGURE = 'SolverLogs.png'
LOGGINGNAME = 'solutionlogs.pkl'
TRAJNAME = 'estimatedtrajectory.pkl'

def make_block():
    block = Block()
    block.Finalize()
    block.terrain = SemiparametricContactModel.FlatSurfaceWithRBFKernel(friction = 0.5, length_scale = 0.1, reg=0.01)
    return block

def get_data(sourcepart):
    """Load the data from the directory sourcepart"""
    file = os.path.join(SOURCEBASE, sourcepart, SOURCEFILE)
    data = utils.load(utils.FindResource(file))
    return data

def make_estimator(data):
    block = make_block()
    traj = ce.ContactEstimationTrajectory(block, data['state'][:, 0])    
    estimator = ce.ContactModelEstimator(traj, horizon=1)
    # Set the costs appropriately
    estimator.forcecost = 1e-1
    estimator.relaxedcost = 1e4
    estimator.distancecost = 1e-3
    estimator.frictioncost = 1e-3
    estimator.useSnoptSolver()
    estimator.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    return estimator

def run_estimation(filepart):
    data = get_data(filepart)
    estimator = make_estimator(data)
    logger = OptimizationLogger(estimator)
    # Run the initial point
    estimator.create_estimator()
    print(f'Contact estimation at timestep {data["time"][0]:.2f}:', end='', flush=True)
    result = estimator.solve()
    print(f" Solved successfully? {result.is_success()}")
    estimator.update_trajectory(data['time'][0], result)
    logger.log(result)
    # Loop over each part of the contact estimation problem
    for t, x, u in zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 0:].T):
        print(f'Contact estimation at timestep {t:.2f}:', end='', flush=True)
        estimator.traj.append_sample(t, x, u)
        estimator.create_estimator()
        result = estimator.solve()
        print(f" Solved successfully? {result.is_success()}")
        estimator.update_trajectory(t, result)
        logger.log(result)
    # Save the contact estimation trajectory
    estimator.traj.saveContactTrajectory(os.path.join(SAVEDIR, filepart, TRAJNAME))
    # Plot the overall results
    plotter = ce.ContactEstimationPlotter(estimator.traj)
#    plotter.plot(show=True)
    plotter.plot(show=False, savename=os.path.join(SAVEDIR, filepart, FIGURENAME))
    logger.plot(show=False, savename=os.path.join(SAVEDIR, filepart, LOGFIGURE))
    logger.save(filename = os.path.join(SAVEDIR, filepart, LOGGINGNAME))
    plt.close('all')

def replot_logs(filepart):
    logger = OptimizationLogger.load(os.path.join(SAVEDIR, filepart, LOGGINGNAME))
    logger.plot(show=True)

def open_logs(filepart):
    logger = OptimizationLogger.load(os.path.join(SAVEDIR, filepart, LOGGINGNAME))
    print("Hello!")

if __name__ == '__main__':
    run_estimation('steppedterrain')
    run_estimation('flatterrain')
    run_estimation('lowfriction')
    run_estimation('highfriction')
    # replot_logs('flatterrain')
    #open_logs('flatterrain')