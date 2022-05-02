import os
import pycito.utilities as utils
from pydrake.all import PiecewisePolynomial as pp
from pycito.systems.A1.a1 import A1VirtualBase

data = utils.load(os.path.join('data','a1','a1_step.pkl'))

xtraj = pp.FirstOrderHold(data['time'], data['state'])
utraj = pp.ZeroOrderHold(data['time'], data['control'])
ftraj = pp.ZeroOrderHold(data['time'], data['force'])
jtraj = pp.ZeroOrderHold(data['time'], data['jointlimit'])

a1 = A1VirtualBase()
a1.Finalize()
a1.plot_trajectories(xtraj, utraj, ftraj, jtraj, show=False, savename=os.path.join('data', 'a1', 'a1_step.png'))
a1.plot_foot_trajectory(xtraj, show=False, savename=os.path.join('data','a1','a1_step_foot.png'))