import os
import numpy as np
import matplotlib.pyplot as plt

from pycito.controller.optimization import OptimizationLogger

SOURCE = os.path.join('examples','a1','estimation_in_the_loop','mpc','flatterrain','3m')
file = os.path.join(SOURCE, 'mpclogs.pkl')

logs = OptimizationLogger.load(file).logs
times = np.array([log['solvetime'] for log in logs])

print(f"Mean solve time: {np.mean(times):.2f} +/- {np.std(times):.2f}")

fig, ax = plt.subplots(1,1)
ax.hist(times, bins=15)
ax.set_xlabel('Solve Time (s)')
ax.set_ylabel("Frequency")
ax.set_title('MPC Solution times')

fig.savefig(os.path.join(SOURCE, 'mpc_solve_times.png'), dpi=fig.dpi, bbox_inches='tight')