import os
import numpy as np
import pycito.utilities as utils
import matplotlib.pyplot as plt

reffile = os.path.join('data','a1','a1_step.pkl')
simfile = os.path.join('examples','a1','simulation_tests','timestepping_2','simdata.pkl')

ref = utils.load(utils.FindResource(reffile))
sim = utils.load(utils.FindResource(simfile))

sim['time'] = sim['time'][:-1]
sim['state'] = sim['state'][:, :-1]

nQ = int(sim['state'].shape[0]/2)

pos_err = ref['state'][:nQ,:] - sim['state'][:nQ, :]
pos_err = np.sum(pos_err **2, axis=0)
vel_err = ref['state'][nQ:,:] - sim['state'][nQ:, :]
vel_err = np.sum(vel_err**2, axis=0)

fig, axs = plt.subplots(2,1)
axs[0].plot(ref['time'], pos_err, 'o-', linewidth=1.5)
axs[1].plot(ref['time'], vel_err, 'o-', linewidth=1.5)
axs[0].set_ylabel('Position')
axs[0].set_title('Tracking Error')
axs[1].set_ylabel('Velocity')
axs[1].set_xlabel('Time (s)')
plt.show()