import os
from collections import Counter
from pycito.controller.optimization import OptimizationLogger

SOURCE = os.path.join('examples','a1','estimation_in_the_loop','offline_estimation','flatterrain','solutionlogs.pkl')

logger = OptimizationLogger.load(SOURCE)

codes = Counter([log['exitcode'] for log in logger.logs])
for key, value in codes.items():
    print(f"SNOPT Exited with code {key} {value} times")

