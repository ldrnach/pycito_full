from pydrake.all import PiecewisePolynomial
import numpy as np
import pycito.utilities as utils
from pycito.systems.A1.a1 import A1VirtualBase
import os

def reconstruct_trajectories(solndict):
    t = solndict['time']
    xtraj = reconstruct_state(t, solndict['state'], solndict['state_order'], solndict['timesteps'].size)
    ltraj = PiecewisePolynomial.ZeroOrderHold(t, solndict['force'])
    jltraj = PiecewisePolynomial.ZeroOrderHold(t, solndict['jointlimit'])
    t_ = np.cumsum(solndict['timesteps'])
    t_ = np.concatenate((np.zeros((1,)), t_), axis=0)
    utraj = PiecewisePolynomial.ZeroOrderHold(t_, solndict['control'])
    return xtraj, utraj, ltraj, jltraj

def reconstruct_state(t, x, order, N):
    xtraj = PiecewisePolynomial.LagrangeInterpolatingPolynomial(t[0:order+2], x[:, 0:order+2])
    for n in range(1, N):
        start = n * (order + 1)
        stop = (n+1)*(order+1) +1
        piece = PiecewisePolynomial.LagrangeInterpolatingPolynomial(t[start:stop], x[:, start:stop])
        xtraj.ConcatenateInTime(piece)
    return xtraj

if __name__ == '__main__':
    a1 = A1VirtualBase()
    a1.Finalize()
    dir = os.path.join('data','a1_walking','collocation','weight_1')
    file = os.path.join(dir, 'a1_walking_collocation_results.pkl')
    data = utils.load(file)
    xtraj, utraj, ltraj, jltraj = reconstruct_trajectories(data)
    a1.plot_trajectories(xtraj, utraj, ltraj, jltraj, samples=10000, show=False, savename=os.path.join(dir,'A1WalkingCollocation.png'))
    # Visualize the motion
    a1.visualize(xtraj)
