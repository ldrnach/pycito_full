import os
import numpy as np
import matplotlib.pyplot as plt
from pycito.utilities import load
from pycito.systems.A1.a1 import A1VirtualBase

from pydrake.all import PiecewisePolynomial

SOURCE = os.path.join('data', 'a1_experiment','a1_hardware_data_2.pkl')
data = load(SOURCE)

a1 = A1VirtualBase()
a1.Finalize()

nq = a1.multibody.num_positions()

pos = data['state'][0:nq,:]
t = np.squeeze(data['time'])

# Estimate the velocity with finite difference method
dt = np.diff(t)
vel_est = np.diff(pos)/dt

state_diff = np.copy(data['state'])
state_diff[nq:, 1:] = vel_est

x_est = PiecewisePolynomial.FirstOrderHold(t, state_diff)
x_traj = PiecewisePolynomial.FirstOrderHold(t, data['state'])

a1.plot_state_trajectory(x_est, show=False)
a1.plot_state_trajectory(x_traj, show=False)

plt.show()