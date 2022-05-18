import os, copy
import numpy as np
import pycito.utilities as utils
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.controller import contactestimator as ce
import pycito.systems.contactmodel as cm
import pycito.systems.kernels as kernels

HORIZON = 1
SOURCE = os.path.join('examples','a1','simulation_tests','fullstep','timestepping','simdata.pkl')
TARGET = os.path.join('examples','a1','estimation_in_the_loop','offline_estimation','singlestep',f'N{HORIZON}')
TRAJNAME = 'estimatedtrajectory.pkl'
FIGURENAME = 'EstimationResults.png'
LOGFIGURE = 'SolverLogs.png'
LOGGINGNAME = 'solutionlogs.pkl'

def make_a1():
    a1 = A1VirtualBase()
    kernel = kernels.RegularizedPseudoHuberKernel(length_scale = np.array([0.01, 0.01, np.inf]), delta = 0.1, noise = 0.01)
    a1.terrain = cm.SemiparametricContactModel(
        surface = cm.SemiparametricModel(cm.FlatModel(location = 0., direction = np.array([0., 0., 1.0])), kernel = kernel),
        friction = cm.SemiparametricModel(cm.ConstantModel(const = 1.0), kernel = copy.deepcopy(kernel))
    )
    a1.Finalize()
    return a1

def make_estimator(data):
    a1 = make_a1()
    traj = ce.ContactEstimationTrajectory(a1, data['state'][:,0])
    estimator = ce.ContactModelEstimator(traj, horizon=HORIZON)
    # Set the costs appropriately
    estimator.forcecost = 1e0
    estimator.relaxedcost = 1e3
    estimator.distancecost = 1e-1
    estimator.frictioncost = 1e1
    estimator.useSnoptSolver()
    estimator.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    estimator.enableLogging()
    return estimator

def run_estimator():
    data = utils.load(SOURCE)
    estimator = make_estimator(data)
    # Run the initial point
    estimator.create_estimator()
    print(f'Contact estimation at timestep {data["time"][0]:.2f}:', end='', flush=True)
    result = estimator.solve()
    print(f" Solved successfully? {result.is_success()}")
    estimator.update_trajectory(data['time'][0], result)
    # Loop over each part of the contact estimation problem
    for t, x, u in zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 0:].T):
        print(f'Contact estimation at timestep {t:.2f}:', end='', flush=True)
        estimator.traj.append_sample(t, x, u)
        estimator.create_estimator()
        result = estimator.solve()
        print(f" Solved successfully? {result.is_success()}")
        estimator.update_trajectory(t, result)
    # Save the contact estimation trajectory
    estimator.traj.saveContactTrajectory(os.path.join(TARGET, TRAJNAME))
    # Plot the overall results
    plotter = ce.ContactEstimationPlotter(estimator.traj)
    plotter.plot(show=False, savename=os.path.join(TARGET, FIGURENAME))
    estimator.logger.plot(show=False, savename=os.path.join(TARGET, LOGFIGURE))
    estimator.logger.save(filename = os.path.join(TARGET, LOGGINGNAME))

if __name__ == '__main__':
    run_estimator()