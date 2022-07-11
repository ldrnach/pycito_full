import numpy as np
import os
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.systems.visualization import Visualizer
import pycito.utilities as utils

from pydrake.all import RigidTransform, SpatialInertia, UnitInertia, PrismaticJoint, BallRpyJoint, PiecewisePolynomial

SOURCE = os.path.join('examples','a1','estimation_in_the_loop','mpc','lowfriction','3m','simdata.pkl')
TERRAIN = os.path.join("pycito","systems","urdf","slipterrain.urdf")
A1_URDF = os.path.join("pycito","systems","A1","A1_description","urdf","a1_no_collision.urdf")

def weld_base_to_world(visualizer, model_index):
    body_index = visualizer.plant.GetBodyIndices(model_index)
    body_frame = visualizer.plant.get_body(body_index[0]).body_frame()
    visualizer.plant.WeldFrames(visualizer.plant.world_frame(), body_frame, RigidTransform(np.zeros(3,)))
    return visualizer

def add_terrain(visualizer):
    visualizer.addModelFromFile(utils.FindResource(TERRAIN))
    visualizer = weld_base_to_world(visualizer, visualizer.model_index[-1])
    return visualizer

def create_a1_visualizer():
    vis = Visualizer(utils.FindResource(A1_URDF))
    # Add in virtual joints to represent the floating base
    zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
    # Create virtual, zero-mass
    xlink = vis.plant.AddRigidBody('xlink', vis.model_index, zeroinertia)
    ylink = vis.plant.AddRigidBody('ylink', vis.model_index, zeroinertia)
    zlink = vis.plant.AddRigidBody('zlink', vis.model_index, zeroinertia)
    # Create the translational and rotational joints
    xtrans = PrismaticJoint("xtranslation", vis.plant.world_frame(), xlink.body_frame(), [1., 0., 0.])
    ytrans = PrismaticJoint("ytranslation", xlink.body_frame(), ylink.body_frame(), [0., 1., 0.])
    ztrans = PrismaticJoint("ztranslation", ylink.body_frame(), zlink.body_frame(), [0., 0., 1.])
    rpyrotation = BallRpyJoint("baseorientation", zlink.body_frame(), vis.plant.GetBodyByName('base').body_frame())
    # Add the joints to the multibody plant
    vis.plant.AddJoint(xtrans)
    vis.plant.AddJoint(ytrans)
    vis.plant.AddJoint(ztrans)
    vis.plant.AddJoint(rpyrotation)
    return vis

def main():
    vis = create_a1_visualizer()
    vis = add_terrain(vis)
    data = utils.load(SOURCE)
    xtraj = PiecewisePolynomial.FirstOrderHold(data['time'], data['state'])
    vis.visualize_trajectory(xtraj)

if __name__ == '__main__':
    main()