import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from pycito.controller.contactestimator import ContactEstimationTrajectory
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.systems.contactmodel import SemiparametricContactModel
import pycito.utilities as utils

DIR = os.path.join('hardwareinterface','debug','online')
LCM_SOURCE = os.path.join(DIR, 'estimator_debug.pkl')
EST_SOURCE = os.path.join(DIR, 'contact_trajectory.pkl')

a1 = A1VirtualBase()
a1.terrain = SemiparametricContactModel.FlatSurfaceWithRBFKernel()
a1.Finalize()

traj = ContactEstimationTrajectory.load(a1, EST_SOURCE)
lcm_data = utils.load(LCM_SOURCE)

lcm_foot = np.column_stack(lcm_data['wbc_lcm_data']['foot_pos'])
traj_foot = np.column_stack([np.ravel(cpt.T) for cpt in traj._contactpoints])
traj_foot = traj_foot[:, 1:]

print(f"LCM Foot trajectory size: {lcm_foot.shape}")
print(f"Estimator foot trajectory size: {traj_foot.shape}")

# Get the time axes
lcm_time = np.asarray(lcm_data['wbc_lcm_data']['lcm_timestamp'])
traj_time = np.asarray(traj._time[1:])

print(f"LCM Average sampling time: {np.mean(np.diff(lcm_time))} +- {np.std(np.diff(lcm_time))}")
print(f"Estimator Average sampling time: {np.mean(np.diff(traj_time))} += {np.std(np.diff(traj_time))}")
# Plot the foot trajectories
fig, axs = plt.subplots(3, 1)
for k in range(3):
    for n in range(4):
        axs[k].plot(lcm_time, lcm_foot[3*n + k, :], linewidth=1.5, label=f"Foot {n}")
axs[0].set_ylabel('Horizontal (m)')
axs[1].set_ylabel('Lateral (m)')
axs[2].set_ylabel('Vertical (m)')
axs[2].set_xlabel('LCM Time (s)')
axs[0].set_title('LCM Foot Trajectories')
axs[0].legend(frameon=False)

# Plot the estimator foot trajectory
fig, axs = plt.subplots(3,1)
for k in range(3):
    for n in range(4):
        axs[k].plot(traj_time, traj_foot[3*n + k, :], linewidth=1.5, label=f"Foot {n}")
axs[0].set_ylabel('Horizontal (m)')
axs[1].set_ylabel('Lateral (m)')
axs[2].set_ylabel('Vertical (m)')
axs[2].set_xlabel('Estimator Time (s)')
axs[0].set_title('Estimator Foot Trajectories')
axs[0].legend(frameon=False)

#plt.show()

print('Running cross-correlation analysis')
DOWNSAMPLING = 50
indices = np.arange(0, lcm_time.size+1, DOWNSAMPLING)

lcm_ds_time = lcm_time[indices]
print(f'After downsampling, LCM average sampling time: {np.mean(np.diff(lcm_ds_time))} +- {np.std(np.diff(lcm_ds_time))}')
lcm_ds_feet = lcm_foot[:, indices]

xcorr = []
for k in range(12):
    xcorr.append(signal.correlate(traj_foot[k,:], lcm_ds_feet[k,:]))

xcorr = np.row_stack(xcorr)
lags = np.arange(-lcm_ds_feet.shape[1]+2, traj_foot.shape[1])


# Plot the cross correlation 
fig, axs = plt.subplots(3,1)
for k in range(3):
    for n in range(4):
        axs[k].plot(lags, xcorr[3*n + k, 1:], linewidth=1.5, label=f"Foot {n}")
axs[0].set_ylabel('Horizontal (m)')
axs[1].set_ylabel('Lateral (m)')
axs[2].set_ylabel('Vertical (m)')
axs[2].set_xlabel('Lag (0.01s)')
axs[0].set_title('Cross-Correlation Functions')
axs[0].legend(frameon=False)

# Print the maximum lags
shift = [lags[np.argmax(xcorr[k,:])] for k in range(12)]
print(f'Maximum correlation at lag: {shift}')
print(f"Average signal shift: {np.mean(shift) * 0.01}")

plt.show()