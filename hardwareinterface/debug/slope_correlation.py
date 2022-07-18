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
axs[1].plot(lags, xcorr, linewidth=1.5)
axs[0].set_ylabel('Pitch (rad)')
axs[0].set_xlabel('Time (s)')
axs[1].set_ylabel('Cross-Correlation')
axs[1].set_xlabel('Lag')
axs[0].legend(frameon=False)

plt.show()