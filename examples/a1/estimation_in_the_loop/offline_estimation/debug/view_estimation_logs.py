import os, copy
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pycito.controller.optimization import OptimizationLogger
from pycito.systems.A1.a1 import A1VirtualBase
import pycito.systems.kernels as kernels
import pycito.systems.contactmodel as cm
from pycito.controller.contactestimator import ContactEstimationTrajectory

SOURCE = os.path.join('examples','a1','estimation_in_the_loop','offline_estimation','singlestep','N1','testing','linearrelaxedcost','solutionlogs.pkl')
TRAJSOURCE = os.path.join('examples','a1','estimation_in_the_loop', 'offline_estimation','singlestep','N1','testing','linearrelaxedcost','estimatedtrajectory.pkl')

a1 = A1VirtualBase()
kernel = kernels.WhiteNoiseKernel(noise=1)
#kernel = kernels.RegularizedPseudoHuberKernel(length_scale = np.array([0.01, 0.01, np.inf]), delta = 0.1, noise = 0.01)
a1.terrain = cm.SemiparametricContactModel(
    surface = cm.SemiparametricModel(cm.FlatModel(location = 0., direction = np.array([0., 0., 1.0])), kernel = kernel),
    friction = cm.SemiparametricModel(cm.ConstantModel(const = 1.0), kernel = copy.deepcopy(kernel))
)
a1.Finalize()

logger = OptimizationLogger.load(SOURCE)
traj = ContactEstimationTrajectory.load(a1, TRAJSOURCE)

# Calculate dynamics error
dyn_err = np.zeros((traj.num_timesteps,))
for k, (dynamics, forces) in enumerate(zip(traj._dynamics_cstr, traj._forces)):
    err = dynamics[0].dot(forces) - dynamics[1]
    dyn_err[k] = np.max(np.abs(err))
# Calculate distance complementarity
dist_err = np.zeros((traj.num_timesteps,))
for k, (distance, derr, forces) in enumerate(zip(traj._distance_cstr, traj._distance_error, traj._forces)):
    fN = forces[:traj.num_contacts]
    err = (distance + derr) * fN
    dist_err[k] = np.max(np.abs(err))
# Calculate the sliding velocity complementarity
vel_err = np.zeros((traj.num_timesteps,))
for k, (vel, slack, forces) in enumerate(zip(traj._dissipation_cstr, traj._dissipation_slacks, traj._forces)):
    fT = forces[traj.num_contacts:]
    err = (traj._dissipation_matrix.T.dot(slack) + vel) * fT
    vel_err[k] = np.max(np.abs(err))
# Calculate the friction cone error
fric_err = np.zeros((traj.num_timesteps, ))
for k, (slack, fric, ferr, forces) in enumerate(zip(traj._slacks, traj._friction_cstr, traj._friction_error, traj._forces)):
    fN, fT = forces[:traj.num_contacts], forces[traj.num_contacts:]
    err = slack * ((fric + ferr) * fN - traj._D.dot(fT))
    fric_err[k] = np.max(np.abs(err))

fig, axs = plt.subplots(1,1)
axs.plot(dyn_err, linewidth=1.5, label='Dynamics')
axs.plot(dist_err, linewidth=1.5, label='Distance')
axs.plot(vel_err, linewidth=1.5, label='Dissipation')
axs.plot(fric_err, linewidth=1.5, label='Friction')
axs.set_ylabel('Constraint Violation')
axs.set_xlabel('Problem number (N)')
axs.set_yscale('symlog', linthresh=1e-6)
axs.grid()
axs.legend(frameon=False)
plt.show()

codes = Counter([log['exitcode'] for log in logger.logs])
for key, value in codes.items():
    print(f"SNOPT Exited with code {key} {value} times")

