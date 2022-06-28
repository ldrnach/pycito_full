
import os

import numpy as np
import matplotlib.pyplot as plt


from pycito.controller.optimization import OptimizationLogger

SOURCEDIR = os.path.join('examples','a1','estimation_in_the_loop','mpc','flatterrain_relaxed','3m')
SOURCE = os.path.join(SOURCEDIR, 'mpclogs.pkl')


logger = OptimizationLogger.load(SOURCE)

times = np.array([log['solvetime'] for log in logger.logs])

mean = np.mean(times)
std = np.std(times)

plt.hist(times, bins=50)
plt.xlabel('Solve time (s)')
plt.ylabel('Frequency')
plt.grid(False)
plt.text(5, 40, f'mean = {mean:0.2f}\nstd = {std:0.2f}')
plt.savefig(os.path.join(SOURCEDIR, 'mpcsolvetimes.png'))
plt.show()