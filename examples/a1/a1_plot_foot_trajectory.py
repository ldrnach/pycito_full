from pycito.systems.A1.a1 import A1VirtualBase
from pycito.utilities import load, find_filepath_recursive

from pydrake.all import PiecewisePolynomial as pp

import os

# Make a1
a1 = A1VirtualBase()
a1.Finalize()

# Find all such files, and save the results
sourcedir = os.path.join('examples','a1','foot_tracking_gait')
sourcename = 'trajoptresults.pkl'
savename = 'A1OptFootTrajectory.png'
for directory in find_filepath_recursive(sourcedir, sourcename):
    data = load(os.path.join(directory,sourcename))
    print(f"Plotting data for {os.path.join(directory, sourcename)}")
    # Recreate the trajectory
    xtraj = pp.FirstOrderHold(data['time'],data['state'])
    # Plot and save the results.
    a1.plot_foot_trajectory(xtraj, show=False, savename=os.path.join(directory, savename))