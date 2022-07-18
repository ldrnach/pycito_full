import os
import numpy as np

from pycito.systems.A1.a1 import A1VirtualBase
from pycito.controller.contactestimator import ContactEstimationTrajectory
from pycito.controller.optimization import OptimizationLogger
from pycito.systems.contactmodel import SemiparametricContactModel
SOURCE = os.path.join('hardwareinterface','data','online_debug')
LOGS = os.path.join(SOURCE, 'estimator_logs.pkl')
TRAJ = os.path.join(SOURCE, 'contact_trajectory.pkl')

a1 = A1VirtualBase()
a1.terrain = SemiparametricContactModel.FlatSurfaceWithRBFKernel()
a1.Finalize()

logger = OptimizationLogger.load(LOGS)
traj = ContactEstimationTrajectory.load(a1, TRAJ)

print('loaded logger and trajectory')