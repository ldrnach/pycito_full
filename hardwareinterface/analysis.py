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
axs[0].plot([pytime[0], pytime[-1]], [12, 12], 'k--')
axs[0].plot([pytime[0], pytime[-1]], [-12, -12], 'k--')
axs[0].set_ylabel('Pitch (deg)')
axs[0].set_xlabel('Time (s)')

axs[0].legend(frameon=False)

# Plot the foot trajectories as well
feet = np.column_stack(data['wbc_lcm_data']['foot_pos'])
feet_z = feet[[2,5,8,11],:]
feet_time = data['wbc_lcm_data']['lcm_timestamp']
axs[1].plot(feet_time, feet_z.T, linewidth=1.5)
axs[1].set_ylabel('Foot Vertical Position (m)')
axs[1].set_xlabel('Time (s)')
axs[1].set_xlim(axs[0].get_xlim())
fig.set_tight_layout(True)
fig.savefig(TARGET, dpi=fig.dpi)
plt.show()


