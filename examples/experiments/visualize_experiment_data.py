import os

import numpy as np
from pycito.utilities import load
from pycito.systems.A1.a1 import A1VirtualBase

from pydrake.all import PiecewisePolynomial

"""
Notes:
1. Shorten trajectory to the range 2-25s - A1 isn't moving outside this range
2. Consider downsampling to 20Hz (0.05s intervals)
3. Check that the velocities are reasonably accurate.  - mostly OK

"""

SOURCE = os.path.join('data','a1_experiment','a1_hardware_data.pkl')

data = load(SOURCE)

a1 = A1VirtualBase()
a1.Finalize()
xtraj = PiecewisePolynomial.FirstOrderHold(np.squeeze(data['time']), data['state'])
utraj = PiecewisePolynomial.ZeroOrderHold(np.squeeze(data['time']), data['control'])
a1.plot_trajectories(xtraj, utraj, show=True)



A1VirtualBase.visualize(xtraj)