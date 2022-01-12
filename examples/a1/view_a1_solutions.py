"""
Visualize the solutions from A1 trajectory optimization

Luke Drnach
February 2021
"""

from pycito.systems.A1.a1 import A1, A1VirtualBase
import pycito.utilities as utils
from pydrake.all import PiecewisePolynomial

# Make an A1 model
a1 = A1VirtualBase()
a1.Finalize()
# Load the data
file = "data/a1/a1_virtual_static_regularized_opt_06152021.pkl"
data = utils.load(file)
# Convert data to trajectories
xtraj = PiecewisePolynomial.FirstOrderHold(data['time'], data['state'])
utraj = PiecewisePolynomial.FirstOrderHold(data['time'], data['control'])
ltraj = PiecewisePolynomial.FirstOrderHold(data['time'], data['force'])
jltraj = PiecewisePolynomial.FirstOrderHold(data['time'], data['jointlimit'])
# Plot all trajectories
a1.plot_trajectories(xtraj,utraj,ltraj,jltraj)