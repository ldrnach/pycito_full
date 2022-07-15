import os
import numpy as np

from pycito.utilities import load
from pycito.systems.A1.a1 import A1

from pydrake.all import PiecewisePolynomial

a1 = A1()
a1.Finalize()

file = os.path.join('drake_simulation','mpc_walking_sim','simdata.pkl')
data = load(file)

index = np.argmax(data['time'] > 0.6)

xtraj = PiecewisePolynomial.FirstOrderHold(data['time'][:index], data['state'][:, :index])
utraj = PiecewisePolynomial.ZeroOrderHold(data['time'][:index], data['control'][:, :index])

a1.plot_trajectories(xtraj, utraj, savename=os.path.join('drake_simulation','debug','sim.png'))