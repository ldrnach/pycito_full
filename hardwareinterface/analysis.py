import os
import numpy as np
import matplotlib.pyplot as plt

from pycito import utilities as utils

SOURCE = os.path.join('hardwareinterface','data','hardware_test_07_19.pkl')
TARGET = os.path.join('hardwareinterface','figures','hardware_test_07_19_slopes.png')
if not os.path.exists(TARGET):
    os.makedirs(os.path.dirname(TARGET))


data = utils.load(SOURCE)

pypitch = np.array(data['pycito_command']['pitch']) * 180 / np.pi
pytime = np.array(data['pycito_command']['lcm_timestamp'])
mitpitch = np.array(data['pycito_debug']['pitch_mit']) * 180 / np.pi
mittime = np.array(data['pycito_debug']['lcm_timestamp'])

dt = np.diff(pytime)
fs = 1/dt
print(f"Estimator solver time: {np.mean(dt):0.3f}+-{np.std(dt):0.3f}s")
print(f"Estimator solve frequency: {np.mean(fs):0.2f} +- {np.std(fs):0.2f}Hz")

fig, axs = plt.subplots(2,1)
axs[0].plot(mittime, mitpitch, linewidth=1.5, label='MIT')
axs[0].plot(pytime, pypitch, linewidth=1.5, label='PYCITO')
axs[0].set_ylabel('Pitch (deg)')
axs[0].set_xlabel('Time (s)')
fig.set_tight_layout(True)
axs[0].legend(frameon=False)
fig.savefig(TARGET, dpi=fig.dpi)
plt.show()


