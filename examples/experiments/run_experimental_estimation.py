import os, copy
import numpy as np
import matplotlib.pyplot as plt
import pycito.utilities as utils
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.controller import contactestimator as ce
import pycito.systems.contactmodel as cm
import pycito.systems.kernels as kernels

HORIZON = 5
SOURCE = os.path.join('data','a1_experiment','a1_simulation_samples.pkl')
TARGET = os.path.join('examples','experiments','a1_offline_estimation','hardware_simulation','test1')

TRAJNAME = 'estimatedtrajectory.pkl'
FIGURENAME = 'EstimationResults.png'
LOGFIGURE = 'SolverLogs.png'
LOGGINGNAME = 'solutionlogs.pkl'

def make_a1():
    a1 = A1VirtualBase()
    frickernel = kernels.WhiteNoiseKernel(noise=1)
    surfkernel = kernels.RegularizedCenteredLinearKernel(weights = np.diag([0.01, .01, 0.01]), noise = 0.001)
    #kernel = kernels.RegularizedPseudoHuberKernel(length_scale = np.array([0.01, 0.01, np.inf]), delta = 0.1, noise = 0.01)
    a1.terrain = cm.SemiparametricContactModel(
        surface = cm.SemiparametricModel(cm.FlatModel(location = 0.0, direction = np.array([0., 0., 1.0])), kernel = surfkernel),
        friction = cm.SemiparametricModel(cm.ConstantModel(const = 1.0), kernel = frickernel)
    )
    a1.Finalize()
    return a1

def make_estimator(data):
    a1 = make_a1()
    traj = ce.ContactEstimationTrajectory(a1, data['state'][:,0])
    traj._time[0] = data['time'][0]
    estimator = ce.ContactModelEstimator(traj, horizon=HORIZON)
    # Set the costs appropriately
    estimator.forcecost = 1e2
    estimator.relaxedcost = 1e3
    estimator.distancecost = 1
    estimator.frictioncost = 1
    estimator.velocity_scaling = 1e-3
    estimator.force_scaling = 1e2
    estimator.useSnoptSolver()
    estimator.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    estimator.enableLogging()
    return estimator

def calculate_ground_slope(model):
    null = np.zeros((3,))
    grad = model.surface.gradient(null)
    return np.arctan2(-grad[0,0], grad[0,2]) * 180 / np.pi

def plot_ground_slope(slope, data):
    fig, axs = plt.subplots(1,1)
    t = data['time']
    body_pitch = data['state'][4,:] * 180 / np.pi
    axs.plot(t, slope, linewidth=1.5, label='Estimated Ground Slope')
    axs.plot(t, -body_pitch, linewidth=1.5, label='Body Elevation')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Pitch Angle (degrees)')
    axs.legend(frameon=False)
    fig.savefig(os.path.join(TARGET, 'estimatedslope.png'), dpi=fig.dpi, bbox_inches='tight')
    print('Saved pitch figure!')

def run_estimator():
    data = utils.load(SOURCE)
    slope = np.zeros((data['time'].size,))
    estimator = make_estimator(data)
    # Run the initial point
    estimator.create_estimator()
    print(f'Contact estimation at timestep {data["time"][0]:.2f}:', end='', flush=True)
    result = estimator.solve()
    print(f" Solved successfully? {result.is_success()}")
    estimator.update_trajectory(data['time'][0], result)
    model = estimator.get_updated_contact_model(result)
    slope[0] = calculate_ground_slope(model)
    # Loop over each part of the contact estimation problem
    for k, (t, x, u) in enumerate(zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 1:].T)):
        print(f'Contact estimation at timestep {t:.2f}:', end='', flush=True)
        estimator.traj.append_sample(t, x, u)
        estimator.create_estimator()
        result = estimator.solve()
        print(f" Solved successfully? {result.is_success()}")
        estimator.update_trajectory(t, result)
        model = estimator.get_updated_contact_model(result)
        slope[k+1] = calculate_ground_slope(model)
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
    # Plot the ground slope vs the body pitch angle
    plot_ground_slope(slope, data)
    # Save the estimator report
    report = estimator.generate_report()
    report += f'\n\nData Source: {SOURCE}'
    with open(os.path.join(TARGET, 'estimatorreport.txt'), 'w') as output:
        output.write(report)
    

if __name__ == '__main__':
    run_estimator()