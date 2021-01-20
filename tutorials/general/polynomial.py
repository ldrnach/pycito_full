import numpy as np
from pydrake.all import PiecewisePolynomial

# we can make a trajectory using breaks (the timepoints) and samples (the values)
breaks = [0., 0.5, 1.0, 1.5]
samples = np.array([[0., 0.5, 1.2, 1.6],[1., 0.8, 1.2, 0.2]])
traj = PiecewisePolynomial.FirstOrderHold(breaks, samples)

print(f"The trajectory has {traj.get_number_of_segments()} segments")
print(f"The breaks in the trajectory are {traj.get_segment_times()}")
print(f"The values of the trajectory at the breakpoints are {traj.vector_values(breaks)}")


