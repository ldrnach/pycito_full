import os
import numpy as np
import pycito.utilities as utils
import matplotlib.pyplot as plt

from pycito.systems.A1.a1 import A1VirtualBase

reffile = os.path.join('data','a1','ellipse_foot_tracking','fast','fullstep','weight_1e+03','trajoptresults.pkl')
SOURCE = os.path.join('examples','a1','simulation_tests','fullstep','timestepping_cache')
simfile = os.path.join(SOURCE, 'simdata.pkl')


ref = utils.load(utils.FindResource(reffile))
sim = utils.load(utils.FindResource(simfile))

sim['time'] = sim['time'][:-1]
sim['state'] = sim['state'][:, :-1]
# Calculate the errors
nQ = int(sim['state'].shape[0]/2)
state_err = (ref['state'] - sim['state'])**2
base_err = np.sqrt(np.mean(state_err[:6, :], axis=0))
joint_err = np.sqrt(np.mean(state_err[6:nQ,:],axis=0))
bvel_err = np.sqrt(np.mean(state_err[nQ:nQ+6, :], axis=0))
jvel_err = np.sqrt(np.mean(state_err[nQ+6:, :], axis=0))
# Calculate the foot position errors
a1 = A1VirtualBase()
a1.Finalize()

ref_feet = a1.state_to_foot_trajectory(ref['state'])
sim_feet = a1.state_to_foot_trajectory(sim['state'])
labels = ['FR','FL','BR','BL']
feet = []
for ref_foot, sim_foot in zip(ref_feet, sim_feet):
    err = np.mean((ref_foot - sim_foot)**2, axis=0)
    feet.append(np.sqrt(err))

fig, axs = plt.subplots(3,1)
# Configuration
axs[0].plot(ref['time'], base_err, linewidth=1.5, label='base')
axs[0].plot(ref['time'], joint_err, linewidth=1.5, label='joint')
axs[0].set_ylabel('Position')
axs[0].set_title('Root-Mean-Squared Tracking Error')
# Velocity
axs[1].plot(ref['time'], bvel_err, linewidth=1.5, label='base')
axs[1].plot(ref['time'], jvel_err, linewidth=1.5, label='joint')
axs[1].set_ylabel('Velocity')
# Foot Position
for foot, label in zip(feet, labels):
    axs[2].plot(ref['time'], foot, linewidth=1.5, label=label)
axs[2].set_ylabel('Foot position')
axs[2].set_xlabel('Time (s)')
axs[0].legend(frameon=False, ncol=2)
axs[2].legend(frameon=False, ncol=4)
fig.savefig(os.path.join(SOURCE, 'trackingerror.png'), dpi=fig.dpi, bbox_inches='tight')