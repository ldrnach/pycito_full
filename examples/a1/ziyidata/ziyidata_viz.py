import numpy as np
import os
import matplotlib.pyplot as plt

from pycito.systems.A1.a1 import A1  
from pydrake.all import PiecewisePolynomial as pp

# LOAD THE DATA
source = os.path.join('data','a1','ddp_walk_concatenated.txt')
data = np.loadtxt(source, float).T
t = np.linspace(0, 4, data.shape[1])
state = np.zeros_like(data)
# Reshape the data from (position, orientation, joint) to (orientation, position, joint)
state[0:4,:] = data[[6,3,4,5], :]
state[4:7,:] = data[0:3, :]
state[7:19, :] = data[7:19, :]
state[19:22, :] = data[22:25, :]
state[22:25, :] = data[19:22, :]
state[25:, :] = data[25:, :]
print(state[:19, 0])
xtraj = pp.FirstOrderHold(t, state)
a1 = A1()
a1.Finalize()

feet = a1.state_to_foot_trajectory(state)
swingdur = np.zeros((4,))
for k, foot in enumerate(feet):  
    print(f"Mean step length: {(foot[0,-1] - foot[0,0])/4:.2f}")
    print(f"First Step Height: {np.max(foot[2,:100]):.2f} ")
    # Calculating swing duration
    vel = np.abs(np.diff(foot[2,:]))
    swing = vel > 0.001
    swingdur[k] = np.sum(swing) * 0.01 /4 
    print(f"Mean Swing Phase duration: {swingdur[k]:0.2f}")
print(f"Mean stance phase duration: {(1 - np.sum(swingdur))/4:.2f}")

a1.plot_state_trajectory(xtraj, show=False, savename=os.path.join('data','a1','ddp.png'))
a1.plot_foot_trajectory(xtraj, show=False, savename=os.path.join('data','a1','ddp.png'))
#a1.visualize(xtraj)
# print(state.shape)