"""
Description of A1 robot

Updated January 14, 2020
Includes classes for creating A1 MultibodyPlant and TimesteppingMultibodyPlant as well as an A1Visualizer class
"""

# Project Imports
from utilities import FindResource
from systems.timestepping import TimeSteppingMultibodyPlant
from systems.visualization import Visualizer

class A1(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/A1/A1_description/urdf/a1_foot_collision.urdf"):
        # Initialize the time-stepping multibody plant
        super(A1, self).__init__(file=FindResource(urdf_file))

    @staticmethod
    def visualize(trajectory=None):
        vis = Visualizer("systems/A1/A1_description/urdf/a1_no_collision.urdf")
        vis.visualize_trajectory(xtraj=trajectory)

if __name__ == "__main__":
    a1 = A1()
    a1.visualize()
