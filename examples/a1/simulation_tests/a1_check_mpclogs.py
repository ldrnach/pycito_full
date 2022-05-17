import os
import matplotlib.pyplot as plt
from pycito.controller.optimization import OptimizationLogger

SOURCE = os.path.join('examples','a1','simulation_tests','fullstep')
PARTS = ['timestepping','timestepping_cache']
FILE = os.path.join('mpclogs','mpclogs.pkl')

for part in PARTS:
    logger = OptimizationLogger.load(os.path.join(SOURCE, part, FILE))
    logger.logs, logger.guess_logs = logger.guess_logs, logger.logs
    logger.plot_constraints(show=False, savename=os.path.join(SOURCE, part, 'mpclogs', 'InitialGuessConstraints.png'))
    logger.plot_costs(show=False, savename=os.path.join(SOURCE, part,'mpclogs', 'InitialGuessCosts.png'))