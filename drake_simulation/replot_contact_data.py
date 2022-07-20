import os
import numpy as np
import matplotlib.pyplot as plt
from pycito.utilities import load

SOURCEDIR = os.path.join('drake_simulation','mpc_walking_sim','finetimestep_horizon17')
SOURCE = os.path.join(SOURCEDIR, 'simdata.pkl')
FORCE = os.path.join(SOURCEDIR, 'force_truncated.png')
CONTACT = os.path.join(SOURCEDIR, 'contact_results_truncated.png')

data = load(SOURCE)

ctime = np.array(data['contact']['time'])
STOP = 500
fig, axs = plt.subplots(3,1)
for k in range(3):
    for key in ['FR_foot', 'FL_foot','RR_foot','RL_foot']:
        axs[k].plot(ctime[:-STOP], data['contact'][key]['force'][k,:-STOP], linewidth=1.5, label=key)
axs[0].set_ylabel('Force X')
axs[1].set_ylabel('Force Y')
axs[2].set_ylabel('Force Z')
axs[0].legend(frameon=False)
axs[2].set_xlabel('Time (s)')
fig.set_tight_layout(True)

fig.savefig(FORCE, dpi=fig.dpi)

fig, axs = plt.subplots(3,1)
for k, data_key in enumerate(['penetration_depth', 'separation_speed','slip_speed']):
    for key in ['FR_foot', 'FL_foot','RR_foot','RL_foot']:
        axs[k].plot(ctime[:-STOP], data['contact'][key][data_key][:-STOP], linewidth=1.5, label=key)
    axs[k].set_ylabel(data_key)

axs[0].legend(frameon=False)
axs[2].set_xlabel('Time (s)')
fig.set_tight_layout(True)
fig.savefig(CONTACT, dpi=fig.dpi)
plt.show()