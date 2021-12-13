import os
from systems.A1.a1 import A1VirtualBase
import utilities as utils
import numpy as np
from pydrake.all import PiecewisePolynomial as pp
import matplotlib.pyplot as plt

file = os.path.join('data','a1','warmstarts','staticwalking_51.pkl')
data = utils.load(file)
N = data['state'].shape[1]
t = np.linspace(0, 1, N)

a1 = A1VirtualBase()
a1.Finalize()

xtraj = pp.FirstOrderHold(t, data['state'])
utraj = pp.FirstOrderHold(t, data['control'])
ftraj = pp.FirstOrderHold(t, data['force'])
jltraj = pp.FirstOrderHold(t, data['jointlimit'])

a1.plot_trajectories(xtraj, utraj, ftraj, jltraj, show=False)
plt.show()