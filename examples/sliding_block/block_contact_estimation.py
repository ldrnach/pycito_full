import os
import numpy as np
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


def make_estimator(filepart):
    block = make_block()
    data = get_data(filepart)
    traj = ce.ContactEstimationTrajectory(block,data['state'][:, 0])
    for t, x, u in zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 1:].T):
        traj.append_sample(t, x, u)
    # Create an arbitrary estimation problem for now
    start_ptr = 30
    horizon = 5
    subtraj = traj.subset(0, start_ptr)
    estimator = ce.ContactModelEstimator(subtraj, horizon)
    # Setup the solver and estimator problem
    estimator.useSnoptSolver()
    estimator.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    estimator.create_estimator()
    return estimator

def check_program_variables(estimator):
    dvars = estimator.prog.decision_variables()
    names = [dvar.get_name() for dvar in dvars]
    print(names)



if __name__ == '__main__':
    estimator = make_estimator('flatterrain')
    check_program_variables(estimator)