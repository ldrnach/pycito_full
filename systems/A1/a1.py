"""
Description of A1 robot

Luke Drnach
November 5, 2020

Updated January 14, 2020
Includes classes for creating A1 MultibodyPlant and TimesteppingMultibodyPlant as well as an A1Visualizer class
"""
# Library imports
import numpy as np
from pydrake.geometry import DrakeVisualizer
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.sensors import CameraInfo, RbgdSensor

# Project Imports
from utilities import FindResource
from systems.timestepping import TimeSteppingMultibodyPlant

class A1(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/A1/A1_description/urdf/a1_foot_collision.urdf"):
        # Initialize the time-stepping multibody plant
        super(A1, self).__init__(file=FindResource(urdf_file))


class A1Visualizer():
    def __init__(self):
        pass


if __name__ == "__main__":
    pass