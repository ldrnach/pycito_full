"""
Floating Orientation: Simple example to show specifying orientation in Drake using XYZ-fixed angles

Luke Drnach
April 29, 2021
"""
import numpy as np
from systems.visualization import Visualizer
from pydrake.all import PiecewisePolynomial
from trajopt.quatutils import xyz_to_quaternion, rpy_to_quaternion
# Create visualizer for a block example
vis = Visualizer(urdf = "systems/block/urdf/free_block.urdf")
# Now create a sequence of XYZ-Fixed angles and a time axis
xyz = np.zeros((3,301))
# First rotate about x by 90 degrees,then about Y by 90 degrees
xyz[0,0:101] = np.linspace(0, np.pi/2, 101)
xyz[0,101:] = xyz[0,100]
xyz[1, 100:201] = np.linspace(0, np.pi/2, 101)
xyz[1,201:] = xyz[1,200]
xyz[2,200:301] = np.linspace(0, np.pi/2, 101)
# Convert to quaternion
q = np.zeros((4,301))
for n in range(301):
    #q[:,n] = rpy_to_quaternion(xyz[:,n])
    q[:,n] = xyz_to_quaternion(xyz[:,n])

# Create the trajectory
t = np.linspace(0, 3, 301)
p = np.zeros((3,301))
p[-1,:] = 1
x = np.concatenate([q,p], axis=0)
traj = PiecewisePolynomial.FirstOrderHold(t, x)
vis.visualize_trajectory(traj)
