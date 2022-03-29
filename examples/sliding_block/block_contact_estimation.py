import os
import numpy as np
from collections import defaultdict
from pycito.controller import contactestimator as ce
from pycito.systems.block.block import Block
from pycito.systems.contactmodel import SemiparametricContactModel
import pycito.utilities as utils

SOURCEBASE = os.path.join('examples','sliding_block','simulations')
SOURCEFILE = os.path.join('mpc', 'simdata.pkl')

def make_block():
    block = Block()
    block.Finalize()
    block.terrain = SemiparametricContactModel.FlatSurfaceWithRBFKernel(friction = 0.5, length_scale = 0.1)
    return block

def get_data(sourcepart):
    """Load the data from the directory sourcepart"""
    file = os.path.join(SOURCEBASE, sourcepart, SOURCEFILE)
    data = utils.load(utils.FindResource(file))
    return data

def make_estimator(sourcepart):
    data = get_data(sourcepart)
    block = make_block()
    traj = ce.ContactEstimationTrajectory(block, data['state'][:, 0])    
    for t, x, u in zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 1:].T):
        traj.append_sample(t, x, u)
    estimator = ce.ContactModelEstimator(traj, horizon=5)
    estimator.useSnoptSolver()
    estimator.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    return estimator

def run_estimation(filepart):
    data = get_data(filepart)
    estimator = make_estimator(data)
    
    # Loop over each part of the contact estimation problem
    for t, x, u in zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 1:].T):
        estimator.traj.append_sample(t, x, u)
        estimator.create_estimator()
        result = estimator.solve()

def check_plotter(filepart):
    data = get_data(filepart)
    block = make_block()
    traj = ce.ContactEstimationTrajectory(block, data['state'][:, 0])
    for t, x, u in zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 1:].T):
        traj.append_sample(t, x, u)
    plotter = ce.ContactEstimationPlotter(traj)
    plotter.plot(show=True)

def check_program_variables(estimator):
    estimator.create_estimator()
    dvars = estimator.get_decision_variable_dictionary()
    print(dvars.keys())


if __name__ == '__main__':
    #check_plotter('flatterrain')
    estimator = make_estimator('flatterrain')
    check_program_variables(estimator)