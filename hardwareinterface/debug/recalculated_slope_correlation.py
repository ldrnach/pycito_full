import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from pycito.controller.contactestimator import ContactEstimationTrajectory

from hardwareinterface.a1estimatorinterface import A1ContactEstimationInterface

#TODO: Debug and finish this script

DIR = os.path.join('hardwareinterface','debug','online')
LCM_SOURCE = os.path.join(DIR, 'estimator_debug.pkl')
EST_SOURCE = os.path.join(DIR, 'contact_trajectory.pkl')

interface = A1ContactEstimationInterface()
estimator = interface.estimator

estimator.traj = ContactEstimationTrajectory.load(interface.a1, EST_SOURCE)
slope = []
for time in estimator.traj._time:
    model = estimator.get_contact_model_at_time(time)
    slope.append(interface._calculate_ground_slope(model))

print(f'Number of slope samples: {len(slope)}')