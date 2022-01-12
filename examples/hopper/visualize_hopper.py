"""
Visualize hopper trajectories

Luke Drnach
August 25, 2021
"""

from pycito.systems.hopper.hopper import Hopper as hopper
from pycito.systems.visualization import batch_visualize
import os

dir = os.path.join('examples','hopper','reference_linear')
batch_visualize(hopper, dir)
