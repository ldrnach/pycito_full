"""
Visualize hopper trajectories

Luke Drnach
August 25, 2021
"""

from systems.hopper.hopper import Hopper
from systems.visualization import batch_visualize
import os

dir = os.path.join('examples','hopper','reference')
batch_visualize(Hopper, dir)
