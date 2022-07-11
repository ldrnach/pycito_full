
import os

import numpy as np
import matplotlib.pyplot as plt


from pycito.controller.optimization import OptimizationLogger

SOURCEDIR = os.path.join('examples','a1','estimation_in_the_loop','mpc','flatterrain_relaxed','3m','tests_force_5')
SOURCE = os.path.join(SOURCEDIR, 'mpclogs.pkl')


logger = OptimizationLogger.load(SOURCE)

times = np.array([log['solvetime'] for log in logger.logs])
successes = [log['success'] for log in logger.logs]

mean = np.mean(times)
std = np.std(times)

plt.hist(times, bins=50)
plt.xlabel('Solve time (s)')
plt.ylabel('Frequency')
plt.grid(False)
xl, yl = plt.xlim(), plt.ylim()
xq = xl[0] + 0.1 * (xl[1] - xl[0])
yq = yl[0] + 0.75 * (yl[1] - yl[0])

plt.text(xq, yq, f'mean = {mean:0.2f}\nstd = {std:0.2f}')
plt.savefig(os.path.join(SOURCEDIR, 'mpcsolvetimes.png'))
plt.show()

print(f"MPC solved successfully {sum(successes)} of {len(successes)} times ({sum(successes)/len(successes) * 100 :0.2f}%)")
print(f"MPC takes {mean:0.2f} +- {std:0.2f}s to solve")