"""
Classes and methods for creating and visualizing the sliding block

Luke Drnach
January 14, 2021
"""
import numpy as np
from pydrake.all import RigidTransform
# Project-specific imports
from utilities import FindResource
from systems.timestepping import TimeSteppingMultibodyPlant

class Block(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/block/urdf/sliding_block"):

        # Initialize the time-stepping multibody plant
        super(Block, self).__init__(file=FindResource(urdf_file))
        # Weld the center body frame to the world frame
        body_inds = self.multibody.GetBodyIndices(self.model_index)
        base_frame = self.multibody.get_body(body_inds[0]).body_frame()
        self.multibody.WeldFrames(self.multibody.world_frame(), base_frame, RigidTransform())


class BlockVisualizer():
    def __init__(self, plant=None):
        if plant is None:
            self.plant = Block()
        