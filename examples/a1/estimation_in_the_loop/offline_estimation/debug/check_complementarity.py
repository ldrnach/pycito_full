import os
import numpy as np
import matplotlib.pyplot as plt
from pycito.controller.optimization import OptimizationLogger

SOURCE = os.path.join('examples','a1','estimation_in_the_loop','offline_estimation','singlestep','N1','testing','linearrelaxedcost','solutionlogs.pkl')

logs = OptimizationLogger.load(SOURCE).logs  

distance = np.array([log['distance_slack'].dot(log['normal_forces']) for log in logs])
dissipation = np.array([log['dissipation_slack'].dot(log['friction_forces']) for log in logs])
friction = np.array([log['friction_cone_slack'].dot(log['velocity_slacks']) for log in logs])
relax = np.array([log['relaxation'] for log in logs])

fig, axs = plt.subplots(1,1)
axs.plot(relax, linewidth=1.5, label='Relaxation')
axs.plot(distance, linewidth=1.5, label='Distance_Complementarity')
axs.plot(dissipation, linewidth=1.5, label='Dissipation_Complementarity')
axs.plot(friction, linewidth=1.5, label='Friction_Complementarity')

axs.set_ylabel('Violation')
axs.set_xlabel('Problem Number')
axs.legend(frameon = False)

axs.set_yscale('symlog', linthresh=1e-6)

axs.grid()

plt.show()

print('done')