import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from pycito.utilities import load

DIR = os.path.join('hardwareinterface','debug','online')
LCM_SOURCE = os.path.join(DIR, 'estimator_debug.pkl')

lcm_data = load(LCM_SOURCE)

pycito_pitch = np.array(lcm_data['pycito_debug']['pitch_pycito'])
mit_pitch = np.array(lcm_data['pycito_debug']['pitch_mit'])
time = np.array(lcm_data['pycito_debug']['lcm_timestamp'])

print(f"LCM Timesteps: {time.shape}")
print(f"MIT Pitch size: {mit_pitch.shape}")
print(f"Estimator pitch size: {pycito_pitch.shape}")

py_pitch = pycito_pitch[1:]
mit_pitch = mit_pitch[1:]

xcorr = signal.correlate(py_pitch, mit_pitch)
lags = np.arange(-mit_pitch.shape[0]+1, py_pitch.shape[0])
print(f"XCORR size: {xcorr.shape}, Lags: {lags.shape}")
dt_all = np.diff(time)
dt = np.mean(dt_all)
print(f'Average LCM Sampling time : {dt} +- {np.std(dt_all)}')
print(f"Correlation Shift: {lags[np.argmax(xcorr)] * dt} s")

fig, axs = plt.subplots(2,1)
axs[0].plot(time, py_pitch, linewidth=1.5, label='PYCITO')
axs[0].plot(time, mit_pitch, linewidth=1.5, label='MIT')
axs[1].plot(lags * dt, xcorr, linewidth=1.5)
axs[0].set_ylabel('Pitch (rad)')
axs[0].set_xlabel('Time (s)')
axs[1].set_ylabel('Cross-Correlation')
axs[1].set_xlabel(f'Lag (s)')
axs[0].legend(frameon=False)
axs[1].text(-40, 600, f"Correlation shift: {lags[np.argmax(xcorr)] * dt:0.4f} s")
fig.set_tight_layout(True)
plt.show()
fig.savefig(os.path.join(DIR, 'figures','lcm_pitch_correlation.png'), dpi=fig.dpi)

fig, axs = plt.subplots(2,1)
pypitch_rate = np.diff(py_pitch)
mitpitch_rate = np.diff(mit_pitch)
axs[0].plot(time[1:], mitpitch_rate, linewidth=1.5, label='MIT')
axs[0].plot(time[1:], pypitch_rate, linewidth=1.5, label='PYCITO')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Change in slope')

axs[0].legend(frameon=False)
slopestart = dt * np.argmax(np.abs(pypitch_rate) > 0)
axs[0].text(4, 0.03, f'PYCITO first change\n in slope at {slopestart:0.3}s')
fig.set_tight_layout(True)
plt.show()
fig.savefig(os.path.join(DIR, 'figures','slopechange.png'), dpi=fig.dpi)