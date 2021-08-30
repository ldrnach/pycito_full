"""
Visualize hopper trajectories

Luke Drnach
August 25, 2021
"""

from systems.hopper.hopper import Hopper
import utilities as utils
import os
from pydrake.all import PiecewisePolynomial

def main(file):
    # Load the file
    data = utils.load(file)
    # Create a footed hopper 
    hopper = Hopper()
    # Turn the state data into a trajectory
    xtraj = PiecewisePolynomial.FirstOrderHold(data['time'],data['state'])
    # Visualize
    hopper.visualize(xtraj)

if __name__ == "__main__":
    file = os.path.join('examples','hopper','feasible_scaled_moreiter','Slack_0E+00','trajoptresults.pkl')
    main(file)