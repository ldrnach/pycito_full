import numpy as np
from pydrake.all import PiecewisePolynomial, PiecewiseQuaternionSlerp, Quaternion
import matplotlib.pyplot as plt
# we can make a trajectory using breaks (the timepoints) and samples (the values)
breaks = [0., 0.5, 1.0, 1.5]
samples = np.array([[0., 0.5, 1.2, 1.6],[1., 0.8, 1.2, 0.2]])
traj = PiecewisePolynomial.FirstOrderHold(breaks, samples)

print(f"The trajectory has {traj.get_number_of_segments()} segments")
print(f"The breaks in the trajectory are {traj.get_segment_times()}")
print(f"The values of the trajectory at the breakpoints are {traj.vector_values(breaks)}")

# Making a piecewise quaternion slerp trajectory for rotations
quat_samples = np.eye(4)
quaternions = [Quaternion(x) for x in quat_samples]
qtraj = PiecewiseQuaternionSlerp(breaks, quaternions)
print('finished')


# Make and plot a zero order hold
hold = PiecewisePolynomial.ZeroOrderHold([0,1],[[1,2]])
t = np.linspace(0, 1, 1000)
vals = hold.vector_values(t)
plt.plot(t, vals[0,:])
plt.show()