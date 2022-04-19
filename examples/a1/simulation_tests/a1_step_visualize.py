from pycito.systems.A1.a1 import A1VirtualBase
from pycito.utilities import load
from pydrake.all import PiecewisePolynomial as pp

import os

data = load(os.path.join('data','a1','a1_step.pkl'))
traj = pp.FirstOrderHold(data['time'], data['state'])
A1VirtualBase.visualize(traj)