import os
import numpy as np
import pycito.utilities as utils
from pycito.systems.visualization import Visualizer
from pycito.controller.mpc import LinearizedContactTrajectory
from pycito.systems.block.block import Block

from pydrake.all import PiecewisePolynomial, RigidTransform, Box, CoulombFriction

URDF = os.path.join("pycito","systems","block","urdf","sliding_block.urdf")
REFSOURCE = os.path.join('data','slidingblock','block_reference.pkl')
DATASOURCE = os.path.join('data','estimationcontrol','block_results')
COLORS = [np.array([0., 80/255, 239/255, 1.]), 
        np.array([250/255, 104/255, 0/255, 1.]),
        np.array([0, 138/255, 0, 1.])]
TRANSLATIONS = [np.array([0, -1.5, 0]),
                np.array([0, 0, 0]),
                np.array([0, 1.5, 0])]

class BlockEnvironment():
    def __init__(self):
        self.friction = 1
        self.shape = (20, 20, 0.1)
        self.location = (0, 0, -0.05)
        self.goal_location = (5.0, 0.0, 0.5)
        self.goal_shape = (1.0, 1.0, 1.0)

    def addEnvironmentToPlant(self, plant):
        """Adds the environment and the goal position to the plant"""
        box = Box(*self.shape)
        T_BG = RigidTransform(p = np.array(self.location))
        friction = CoulombFriction(
            static_friction = self.friction,
            dynamic_friction = self.friction,
        )
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            T_BG,
            box,
            'ground_collision',
            friction
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            T_BG,
            box,
            'ground_visual',
            np.array([153/255, 102/255, 51/255, 0.9])
        )
        plant = self.addGoal(plant) 
        return plant

    def addGoal(self, plant):
        """Add the target position to the environment"""
        box = Box(self.goal_shape[0], self.goal_shape[1], self.goal_shape[2])
        T_BG = RigidTransform(p = np.array(self.goal_location))
        plant.RegisterVisualGeometry(
            plant.world_body(),
            T_BG,
            box,
            'goal_visual',
            np.array([212/255, 175/255, 55/255, 0.5])
        )
        return plant

class StepEnvironment(BlockEnvironment):
    def __init__(self):
        super().__init__()
        self.shape = (10, 20, 0.1)
        self.location = (-2.5, 0, -0.05)
        self.step_shape = (10, 20, 0.1)
        self.step_location = (7.5, 0, -0.55)


    def addEnvironmentToPlant(self, plant):
        plant = super().addEnvironmentToPlant(plant)
        # Add in the step environment
        box = Box(*self.step_shape)
        T_BG = RigidTransform(p = np.array(self.step_location))
        friction = CoulombFriction(
            static_friction = self.friction,
            dynamic_friction = self.friction,
        )
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            T_BG,
            box,
            'step_collision',
            friction
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            T_BG,
            box,
            'step_visual',
            np.array([153/255, 102/255, 51/255, 0.9])
        )
        return plant

class LowFrictionEnvironment(BlockEnvironment):
    def __init__(self):
        super().__init__()
        self.patch_friction = 0.1
        self.patch_shape = (2.0, 1.0, 0.01)
        self.patch_location = (3.0, 0.0, 0.005)


    def addEnvironmentToPlant(self, plant):
        plant = super().addEnvironmentToPlant(plant)
        # Add the low friction patch
        box = Box(*self.patch_shape)
        T_BG = RigidTransform(p = np.array(self.patch_location))
        friction = CoulombFriction(
            static_friction = self.patch_friction,
            dynamic_friction = self.patch_friction,
        )
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            T_BG,
            box,
            'friction_patch_collision',
            friction
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            T_BG,
            box,
            'friction_patch_visual',
            np.array([0., 0., 0.5, 0.5])
        )
        return plant

def weldbodytoworld(plant, index, translation = np.zeros((3,))):
    body_inds = plant.GetBodyIndices(index)
    base_frame = plant.get_body(body_inds[0]).body_frame()
    plant.WeldFrames(plant.world_frame(), base_frame, RigidTransform(translation))

def get_reference_trajectory():
    plant = Block()
    plant.Finalize()
    reftraj = LinearizedContactTrajectory.load(plant, REFSOURCE) 
    return reftraj._time, reftraj._state

def make_reference_visualization():
    # Create the visualizer
    viz = Visualizer(utils.FindResource(URDF))
    weldbodytoworld(viz.plant, viz.model_index)
    viz.setModelColor(viz.model_index, COLORS[0])
    env = BlockEnvironment()
    viz.plant = env.addEnvironmentToPlant(viz.plant)
    # Get the reference trajectory data
    time, state = get_reference_trajectory()
    # Make the visualization
    xtraj = PiecewisePolynomial.FirstOrderHold(time, state)
    viz.visualize_trajectory(xtraj)


def make_friction_visualization():
    urdf = utils.FindResource(URDF)
    # Create the visualizer
    viz = Visualizer(urdf)
    # Add two extra blocks - MPC and CAMPC
    viz.addModelFromFile(urdf, name = 'MPC')
    viz.addModelFromFile(urdf, name = 'CAMPC')
    for ind, color, trans in zip(viz.model_index, COLORS, TRANSLATIONS):
        weldbodytoworld(viz.plant, ind, trans)
        viz.setModelColor(ind, color)
    # Create the environment
    env = LowFrictionEnvironment()
    env.patch_shape = (2.0, 4.0, 0.01)
    env.goal_shape = (1.0, 5.0, 1.0)
    env.goal_location  = (5.0, 0., 0.5)
    viz.plant = env.addEnvironmentToPlant(viz.plant)
    # Get the data
    mpc = utils.load(os.path.join(DATASOURCE,'low_friction','mpcsim.pkl'))
    campc = utils.load(os.path.join(DATASOURCE, 'low_friction', 'campcsim.pkl'))
    t, x = get_reference_trajectory()
    fullstate = np.row_stack([x[:2,:151], mpc['state'][:2,:], campc['state'][:2,:], x[2:,:151], mpc['state'][2:,:], campc['state'][2:,:]])
    xtraj = PiecewisePolynomial.FirstOrderHold(t[:151], fullstate)
    viz.visualize_trajectory(xtraj)

def make_step_visualization():
    example = 'stepterrain'
    urdf = utils.FindResource(URDF)
    # Create the visualizer
    viz = Visualizer(urdf)
    # Add two extra blocks - MPC and CAMPC
    viz.addModelFromFile(urdf, name = 'MPC')
    viz.addModelFromFile(urdf, name = 'CAMPC')
    for ind, color, trans in zip(viz.model_index, COLORS, TRANSLATIONS):
        weldbodytoworld(viz.plant, ind, trans)
        viz.setModelColor(ind, color)
    # Create the environment
    env = StepEnvironment()
    env.goal_shape = (1.0, 5.0, 2)
    env.goal_location  = (5.0, 0., 0.5)
    viz.plant = env.addEnvironmentToPlant(viz.plant)
    # Get the data
    mpc = utils.load(os.path.join(DATASOURCE,example,'mpcsim.pkl'))
    campc = utils.load(os.path.join(DATASOURCE, example, 'campcsim.pkl'))
    t, x = get_reference_trajectory()
    fullstate = np.row_stack([x[:2,:151], mpc['state'][:2,:], campc['state'][:2,:], x[2:,:151], mpc['state'][2:,:], campc['state'][2:,:]])
    xtraj = PiecewisePolynomial.FirstOrderHold(t[:151], fullstate)
    viz.visualize_trajectory(xtraj)


if __name__ == '__main__':
    make_step_visualization()