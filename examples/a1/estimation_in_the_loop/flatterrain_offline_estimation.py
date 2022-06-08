import os, copy
import numpy as np
import pycito.utilities as utils
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.controller import contactestimator as ce
import pycito.systems.contactmodel as cm
import pycito.systems.kernels as kernels

HORIZON = 1
SOURCE = os.path.join('examples','a1','simulation_tests','fullstep','timestepping','simdata.pkl')
TARGET = os.path.join('examples','a1','estimation_in_the_loop','offline_estimation','singlestep',f'N{HORIZON}','testing','nonlinear_friction')
TRAJNAME = 'estimatedtrajectory.pkl'
FIGURENAME = 'EstimationResults.png'
LOGFIGURE = 'SolverLogs.png'
LOGGINGNAME = 'solutionlogs.pkl'

def make_a1():
    a1 = A1VirtualBase()
    kernel = kernels.WhiteNoiseKernel(noise=1)
    #kernel = kernels.RegularizedPseudoHuberKernel(length_scale = np.array([0.01, 0.01, np.inf]), delta = 0.1, noise = 0.01)
    a1.terrain = cm.SemiparametricContactModel(
        surface = cm.SemiparametricModel(cm.FlatModel(location = 0.0, direction = np.array([0., 0., 1.0])), kernel = kernel),
        friction = cm.SemiparametricModel(cm.ConstantModel(const = 0.5), kernel = copy.deepcopy(kernel))
    )
    a1.Finalize()
    return a1

def make_estimator(data):
    a1 = make_a1()
    traj = ce.ContactEstimationTrajectory(a1, data['state'][:,0])
    estimator = ce.ContactModelEstimatorNonlinearFrictionCone(traj, horizon=HORIZON)
    # Set the costs appropriately
    estimator.forcecost = 1e0
    estimator.relaxedcost = 1e3
    estimator.distancecost = 1
    estimator.frictioncost = 1
    estimator.velocity_scaling = 1e-2
    estimator.force_scaling = 1e2
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
    for t, x, u in zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 1:].T):
        print(f'Contact estimation at timestep {t:.2f}:', end='', flush=True)
        estimator.traj.append_sample(t, x, u)
        estimator.create_estimator()
        result = estimator.solve()
        print(f" Solved successfully? {result.is_success()}")
        estimator.update_trajectory(t, result)
    # Save the contact estimation trajectory
    estimator.traj.save(os.path.join(TARGET, TRAJNAME))
    # Plot the overall results
    plotter = ce.ContactEstimationPlotter(estimator.traj)
    plotter.plot(show=False, savename=os.path.join(TARGET, FIGURENAME))
    estimator.logger.plot(show=False, savename=os.path.join(TARGET, LOGFIGURE))
    estimator.logger.save(filename = os.path.join(TARGET, LOGGINGNAME))
    # Swap the solution and guess logs
    estimator.logger.logs, estimator.logger.guess_logs = estimator.logger.guess_logs, estimator.logger.logs
    estimator.logger.plot_costs(show=False, savename=os.path.join(TARGET, 'GuessCosts' + LOGFIGURE))
    estimator.logger.plot_constraints(show=False, savename=os.path.join(TARGET, 'GuessConstraints' + LOGFIGURE))
    
    # # Run a polishing step using the ContactModelRectifer
    # rectifier = ce.EstimatedContactModelRectifier(estimator.traj, surf_max = 10, fric_max=2)
    # rectifier.useSnoptSolver()
    # rectifier.setSolverOptions({'Major feasibility tolerance':1e-6,
    #                             'Major optimality tolerance': 1e-6})
    # model = rectifier.get_global_model()
    # t = estimator.traj.time
    # surf_err = model.surface.model_errors
    # fric_err = model.friction.model_errors
    

if __name__ == '__main__':
    run_estimator()