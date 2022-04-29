from pycito.systems.A1.a1 import A1VirtualBase

from pycito.utilities import load, FindResource

from pydrake.all import PiecewisePolynomial as pp

import os

filename = os.path.join('examples','a1','simulation_tests','timestepping','simdata.pkl')

data = load(FindResource(filename))
print(data.keys())
xtraj = pp.FirstOrderHold(data['time'],data['state'])
A1VirtualBase.visualize(xtraj)

refname = os.path.join('')