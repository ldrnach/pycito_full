import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from pycito.controller.contactestimator import ContactEstimationTrajectory
from pycito.controller.optimization import OptimizationLogger
from pycito.utilities import load

from hardwareinterface.a1estimatorinterface import A1ContactEstimationInterface

#TODO: Debug and finish this script

DIR = os.path.join('hardwareinterface','debug','online')
LCM_SOURCE = os.path.join(DIR, 'estimator_debug.pkl')
EST_SOURCE = os.path.join(DIR, 'contact_trajectory.pkl')
OPT_SOURCE = os.path.join(DIR, 'estimator_logs.pkl')

logs = OptimizationLogger.load(OPT_SOURCE).logs

interface = A1ContactEstimationInterface()
estimator = interface.estimator
interface.logging_enabled = False
estimator.traj = ContactEstimationTrajectory.load(interface.a1, EST_SOURCE)
slope = []
opt_slope = []


for time, log in zip(estimator.traj._time[1:], logs):
    model = estimator.get_contact_model_at_time(time)
    slope.append(interface._calculate_ground_slope(model))
    model.surface._kernel_weights = log['distance_weights']
    opt_slope.append(interface._calculate_ground_slope(model))
slope = np.array(slope)
opt_slope = np.array(opt_slope)

time = estimator.traj._time[1:]
fig, axs = plt.subplots(2,1)
axs[0].plot(time[:-1], slope, linewidth=1.5, label='Estimator Stored Slope')
axs[0].plot(time[:-1], opt_slope, linewidth=1.5, label='Optimization Logged Slope')
axs[0].set_ylabel('Slope (rad)')
axs[0].set_xlabel('Time (s)')
axs[0].legend(frameon=False)

xcorr = signal.correlate(slope, opt_slope)
lags = np.arange(-opt_slope.shape[0]+1, slope.shape[0])
dt = np.mean(np.diff(time))

axs[1].plot(lags *dt, xcorr, linewidth=1.5)
axs[1].set_xlabel('Lag (s)')
axs[1].set_ylabel('Cross-Correlation')
axs[1].text(-40, 5, f"Signal Lag: {dt * lags[np.argmax(xcorr)]:0.3f}s")
fig.set_tight_layout(True)
fig.savefig(os.path.join(DIR, 'figures', 'optimizationcheck.png'), dpi=fig.dpi)
plt.show()