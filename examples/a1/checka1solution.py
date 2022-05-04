import os
import pycito.utilities as utils
from pydrake.all import PiecewisePolynomial as pp
from pycito.systems.A1.a1 import A1VirtualBase

FILE = os.path.join('examples','a1','foot_tracking_gait','first_step','weight_1e+03','trajoptresults.pkl')
data = utils.load(FILE)
print('dataloaded')

a1 = A1VirtualBase()
a1.Finalize()

f = pp.ZeroOrderHold(data['time'], data['force'])
a1.plot_force_trajectory(f, show=True)