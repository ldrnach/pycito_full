import os
from pycito.controller.mpc import LinearizedContactTrajectory
from pycito.systems.A1.a1 import A1VirtualBase
from pydrake.all import PiecewisePolynomial

SOURCEDIR = os.path.join('data','a1','reference','symmetric','3m')
SOURCE = os.path.join(SOURCEDIR, 'reftraj.pkl')

plant = A1VirtualBase()
plant.terrain.friction = 1.0
plant.Finalize()

traj = LinearizedContactTrajectory.loadLinearizedTrajectory(plant, SOURCE)

xtraj = PiecewisePolynomial.FirstOrderHold(traj._time, traj._state)
utraj = PiecewisePolynomial.ZeroOrderHold(traj._time, traj._control)
ftraj = PiecewisePolynomial.ZeroOrderHold(traj._time, traj._force)
jltraj = PiecewisePolynomial.ZeroOrderHold(traj._time, traj._jlimit)

plant.plot_trajectories(xtraj, utraj, ftraj, jltraj, show=False, savename=os.path.join(SOURCEDIR, 'traj.png'))
plant.visualize(xtraj)