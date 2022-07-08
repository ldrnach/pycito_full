import os, copy
import numpy as np
import matplotlib.pyplot as plt
import pycito.utilities as utils
from pycito.controller.contactestimator import ContactEstimationTrajectory
from pycito.controller.optimization import OptimizationLogger
from pycito.systems.A1.a1 import A1VirtualBase
import pycito.systems.contactmodel as cm
import pycito.systems.kernels as kernels

SOURCEDIR = os.path.join('examples','experiments','a1_offline_estimation','hardware_tuning','test3')
DATASOURCE = os.path.join('data','a1_experiment','a1_hardware_samples.pkl')
SOURCE = os.path.join(SOURCEDIR, 'estimatedtrajectory.pkl')
LOGS = os.path.join(SOURCEDIR, 'solutionlogs.pkl')



def make_a1():
    a1 = A1VirtualBase()
    frickernel = kernels.WhiteNoiseKernel(noise=1)
    surfkernel = kernels.RegularizedCenteredLinearKernel(weights = np.diag([0.1, 0.1, 0.0]), noise = 0.01)
    a1.terrain = cm.SemiparametricContactModel(
        surface = cm.SemiparametricModel(cm.FlatModel(location = 0.0, direction = np.array([0., 0., 1.0])), kernel = surfkernel),
        friction = cm.SemiparametricModel(cm.ConstantModel(const = 0.0), kernel = frickernel)
    )
    a1.Finalize()
    return a1

def get_body_pitch():
    data = utils.load(DATASOURCE)
    return -data['state'][4,:] * 180 / np.pi

def calculate_ground_slope(model, cpts, weights):
    model = copy.deepcopy(model)
    model.add_samples(cpts, weights)
    dg = model.gradient(np.zeros((3,)))
    return np.arctan2(-dg[0,0], dg[0,2]) * 180 / np.pi

def calculate_ground_slope_arccos(model, cpts, weights):
    model = copy.deepcopy(model)
    model.add_samples(cpts, weights)
    dg = model.gradient(np.zeros((3,)))
    dg_s = np.sqrt(np.sum(dg ** 2))
    return np.arccos(dg[0,2]/dg_s) * 180 /np.pi

def main():
    a1 = make_a1()
    cetraj = ContactEstimationTrajectory.load(a1, SOURCE)
    logs = OptimizationLogger.load(LOGS).logs
    dweights = [log['distance_weights'] for log in logs]
    surf = a1.terrain.surface
    z_null = np.zeros((3,))
    slope = np.zeros((len(dweights,)))
    for k, (cpt, weights) in enumerate(zip(cetraj._contactpoints, dweights)):
        slope[k] = calculate_ground_slope_arccos(surf, cpt, weights)
    # Make a plot of slope against time
    t = np.row_stack(cetraj._time)
    pitch = get_body_pitch()
    plt.plot(t, slope, linewidth=1.5, label='Ground Slope Estimate')
    plt.plot(t, pitch, linewidth=1.5, label='Body Pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Estimated Ground Slope (degrees)')
    plt.legend(frameon=False)
    plt.show()


if __name__ == '__main__':
    main()