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
from systems.visualization import Visualizer
from systems.terrain import FlatTerrain

class Block(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/block/urdf/sliding_block.urdf", terrain=FlatTerrain()):

        # Initialize the time-stepping multibody plant
        super(Block, self).__init__(file=FindResource(urdf_file), terrain=terrain)
        # Weld the center body frame to the world frame
        body_inds = self.multibody.GetBodyIndices(self.model_index)
        base_frame = self.multibody.get_body(body_inds[0]).body_frame()
        self.multibody.WeldFrames(self.multibody.world_frame(), base_frame, RigidTransform())

    @staticmethod
    def visualize(trajectory=None):
        vis = Visualizer("systems/block/urdf/sliding_block.urdf")
        #Weld the center body frame to the world frame
        body_inds = vis.plant.GetBodyIndices(vis.model_index)
        base_frame = vis.plant.get_body(body_inds[0]).body_frame()
        vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())
        # Make the visualization
        vis.visualize_trajectory(trajectory)

if __name__ == "__main__":
    block = Block()
    block.visualize()        