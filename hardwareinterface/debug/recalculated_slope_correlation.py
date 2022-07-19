import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from pycito.controller.contactestimator import ContactEstimationTrajectory
from pycito.utilities import load

from hardwareinterface.a1estimatorinterface import A1ContactEstimationInterface

#TODO: Debug and finish this script

DIR = os.path.join('hardwareinterface','debug','online')
LCM_SOURCE = os.path.join(DIR, 'estimator_debug.pkl')
EST_SOURCE = os.path.join(DIR, 'contact_trajectory.pkl')

interface = A1ContactEstimationInterface()
estimator = interface.estimator
interface.logging_enabled = False
estimator.traj = ContactEstimationTrajectory.load(interface.a1, EST_SOURCE)
slope = []
for time in estimator.traj._time:
    model = estimator.get_contact_model_at_time(time)
    slope.append(interface._calculate_ground_slope(model))
slope = np.array(slope)

lcm_data = load(LCM_SOURCE)

pycito_pitch = np.array(lcm_data['pycito_debug']['pitch_pycito'])[1:]
mit_pitch = np.array(lcm_data['pycito_debug']['pitch_mit'])[1:]
lcm_time = np.array(lcm_data['pycito_debug']['lcm_timestamp'])

dt_est = np.diff(estimator.traj._time)
dt_lcm = np.diff(lcm_time)
print(f"LCM Average sampling interval: {np.mean(dt_lcm)}")
print(f"Estimator average sampling interval: {np.mean(dt_est)}")

# Plot the pitches
fig, axs = plt.subplots(2,1)
axs[0].plot(lcm_time, pycito_pitch, linewidth=1.5, label='pycito pitch lcm')
axs[0].plot(lcm_time, mit_pitch, linewidth=1.5, label='mit pitch lcm')
axs[0].plot(estimator.traj._time, np.asarray(slope), linewidth=1.5, label='pycito pitch estimator')
axs[0].set_ylabel('Slope (rad)')
axs[0].set_xlabel('Time (s)')
axs[0].legend(frameon=False)

# Cross-Correlations
DOWNSAMPLING = 50
indices = np.arange(0, lcm_time.size+1, DOWNSAMPLING)

lcm_ds_time = lcm_time[indices]
lcm_dt = np.mean(np.diff(lcm_ds_time))
print(f'After downsampling, LCM average sampling time: {np.mean(np.diff(lcm_ds_time))} +- {np.std(np.diff(lcm_ds_time))}')
lcm_ds_pypitch = pycito_pitch[indices]
lcm_ds_mitpitch = mit_pitch[indices]

xcorr1 = signal.correlate(lcm_ds_pypitch, slope)
xcorr2 = signal.correlate(lcm_ds_mitpitch, slope)
lags = np.arange(-slope.shape[0]+1, lcm_ds_pypitch.shape[0])

shift1 = lags[np.argmax(xcorr1)] * lcm_dt
shift2 = lags[np.argmax(xcorr2)] * lcm_dt

axs[1].plot(lags*lcm_dt, xcorr1, linewidth=1.5, label=f'pycito lcm - estimator\nShift {shift1:0.3f}s')
axs[1].plot(lags*lcm_dt, xcorr2, linewidth=1.5, label=f'mit lcm - estimator\nShift {shift2:0.3f}s')
axs[1].set_ylabel('Cross Correlation')
axs[1].set_xlabel(f'Lag (s)')
axs[1].legend(frameon=False)
fig.set_tight_layout(True)
plt.show()

fig.savefig(os.path.join(DIR, 'figures', 'recalculated_slope.png'), dpi=fig.dpi)