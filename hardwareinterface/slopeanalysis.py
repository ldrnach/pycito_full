import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pycito import utilities as utils

from pydrake.all import PiecewisePolynomial
EXT = '.pdf'
SOURCE = os.path.join('hardwareinterface','data','hardware_test_07_21.pkl')
SAVEDIR = os.path.join('hardwareinterface','figures')
if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

TARGET = os.path.join(SAVEDIR, 'hardware_test_07_21_slopes' + EXT)

data = utils.load(SOURCE)

pypitch = np.array(data['pycito_command']['pitch']) * 180 / np.pi
pytime = np.array(data['pycito_command']['lcm_timestamp'])
mitpitch = np.array(data['pycito_debug']['pitch_mit']) * 180 / np.pi
mittime = np.array(data['pycito_debug']['lcm_timestamp'])

if 'roll' in data['pycito_command'].keys():
    pyroll = np.array(data['pycito_command']['roll']) *180 /np.pi
    mitroll = np.array(data['pycito_debug']['roll_mit']) * 180 / np.pi
    useroll = True
else:
    useroll = False

dt = np.diff(pytime)
fs = 1/dt
print(f"Estimator solver time: {np.mean(dt):0.3f}+-{np.std(dt):0.3f}s")
print(f"Estimator solve frequency: {np.mean(fs):0.2f} +- {np.std(fs):0.2f}Hz")

fig, axs = plt.subplots(2,1)
axs[0].add_patch(Rectangle(xy=[pytime[0], 12], width=pytime[-1]-pytime[0], height=3, facecolor = [0.5, 0.5, 0.5, 0.5], edgecolor=None))
axs[0].add_patch(Rectangle(xy=[pytime[0], -15], width=pytime[-1]-pytime[0], height=3, facecolor = [0.5, 0.5, 0.5, 0.5], edgecolor=None))
axs[0].plot(mittime, mitpitch, linewidth=1.5, label='MIT')
axs[0].plot(pytime, pypitch, linewidth=1.5, label='PYCITO')

axs[0].set_ylabel('Pitch (deg)')
axs[0].set_xlabel('Time (s)')

axs[0].legend(frameon=False)

# Plot the foot trajectories as well
if useroll:
    axs[1].add_patch(Rectangle(xy=[pytime[0], -15], width=pytime[-1]-pytime[0], height=3, facecolor = [0.5, 0.5, 0.5, 0.5], edgecolor=None))
    axs[1].plot(mittime, mitroll, linewidth=1.5, label='MIT')
    axs[1].plot(pytime, pyroll, linewidth=1.5, label='PYCITO')
    axs[1].set_ylabel('Roll (deg)')
    axs[1].set_xlabel('Time (s)')
else:
    feet = np.column_stack(data['wbc_lcm_data']['foot_pos'])
    feet_z = feet[[2,5,8,11],:]
    feet_time = data['wbc_lcm_data']['lcm_timestamp']
    axs[1].plot(feet_time, feet_z.T, linewidth=1.5)
    axs[1].set_ylabel('Foot Vertical Position (m)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_xlim(axs[0].get_xlim())
fig.set_tight_layout(True)
fig.savefig(TARGET, dpi=fig.dpi)


# Resample the data and to create even trajectories for analysis
py_pitch_poly = PiecewisePolynomial.FirstOrderHold(pytime.tolist(), [pypitch.tolist()])
mit_pitch_poly = PiecewisePolynomial.FirstOrderHold(mittime.tolist(), [mitpitch.tolist()])
start, stop = max(pytime[0], mittime[0]), min(pytime[-1], mittime[-1])
N = min(pytime.shape[0], mittime.shape[0])
times = np.linspace(start, stop, N)

pypitches = np.array([py_pitch_poly.value(time) for time in times])
mitpitches = np.array([mit_pitch_poly.value(time) for time in times])

#RMSE = np.sqrt(np.mean((pypitches - mitpitches)**2))
AE = np.abs(pypitches - mitpitches)

print(f"Pitch Mean Absolute Error: {np.mean(AE):0.2f} degrees")
print(f"Pitch Maximum Absolute Error: {np.max(AE):0.2f} degrees")

if useroll:
    # Resample the data and to create even trajectories for analysis
    py_roll_poly = PiecewisePolynomial.FirstOrderHold(pytime.tolist(), [pyroll.tolist()])
    mit_roll_poly = PiecewisePolynomial.FirstOrderHold(mittime.tolist(), [mitroll.tolist()])
    start, stop = max(pytime[0], mittime[0]), min(pytime[-1], mittime[-1])
    N = min(pytime.shape[0], mittime.shape[0])
    times = np.linspace(start, stop, N)

    pyrolls = np.array([py_roll_poly.value(time) for time in times])
    mitrolls = np.array([mit_roll_poly.value(time) for time in times])

    #RMSE = np.sqrt(np.mean((pyrolls - mitrolls)**2))
    #print(f"Roll Mean Absolute Error: {RMSE:0.2f} degrees")
    rAE = np.abs(pyrolls - mitrolls)
    print(f"Roll Mean Absolute Error: {np.mean(rAE):0.2f} degrees")
    print(f"Roll Max Absolute Error: {np.max(rAE):0.2f} degrees")

plt.show()