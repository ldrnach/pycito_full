import os
import numpy as np
import pycito.utilities as utils
from pycito.systems.visualization import Visualizer

from pydrake.all import PiecewisePolynomial, SpatialInertia, UnitInertia, PrismaticJoint, BallRpyJoint

REFSOURCE = os.path.join('data','a1','reference','3m','reftraj.pkl')
SIMSOURCE = os.path.join('examples','a1','estimation_in_the_loop','mpc','flatterrain','3m','simdata.pkl')

def add_a1_to_visualizer(vis = None):
    """Add A1 model to the visualizer"""
    # First add the floating base model
    file = utils.FindResource(os.path.join("systems","A1","A1_description","urdf","a1_no_collision.urdf"))
    if vis is None:
        vis = Visualizer(file)
        vis.model_index = [vis.model_index]
    else:
        vis.addModelFromFile(file, name=f'A1({len(vis.model_index) + 1})')
    
    # Add virtual joints to represent the floating base
    zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
    # Create virtual, zero-mass
    xlink = vis.plant.AddRigidBody('xlink', vis.model_index[-1], zeroinertia)
    ylink = vis.plant.AddRigidBody('ylink', vis.model_index[-1], zeroinertia)
    zlink = vis.plant.AddRigidBody('zlink', vis.model_index[-1], zeroinertia)
    # Create the translational and rotational joints
    xtrans = PrismaticJoint("xtranslation", vis.plant.world_frame(), xlink.body_frame(), [1., 0., 0.])
    ytrans = PrismaticJoint("ytranslation", xlink.body_frame(), ylink.body_frame(), [0., 1., 0.])
    ztrans = PrismaticJoint("ztranslation", ylink.body_frame(), zlink.body_frame(), [0., 0., 1.])
    rpyrotation = BallRpyJoint("baseorientation", zlink.body_frame(), vis.plant.GetBodyByName('base', vis.model_index[-1]).body_frame())
    # Add the joints to the multibody plant
    vis.plant.AddJoint(xtrans)
    vis.plant.AddJoint(ytrans)
    vis.plant.AddJoint(ztrans)
    vis.plant.AddJoint(rpyrotation)
    return vis

def set_model_color(vis, model_index, color=None):
    """Changes the colors of the body in the visualizer"""
    if color is None:
        return vis
    for body_ind  in vis.plant.GetBodyIndices(model_index):
        vis.setBodyColor(body_ind, color)
    return vis

def make_a1_multivisualizer(datasets, colors):
    vis = None
    postraj = []
    veltraj = []
    nQ = int(datasets[0]['state'].shape[0]/2)
    for dataset, color in zip(datasets, colors):
        vis = add_a1_to_visualizer(vis)
        vis = set_model_color(vis, vis.model_index[-1], color)
        postraj.append(dataset['state'][:nQ, :])
        veltraj.append(dataset['state'][nQ:, :])
    time = np.squeeze(datasets[1]['time'])
    postraj = np.concatenate(postraj, axis=0)
    veltraj = np.concatenate(veltraj, axis=0)
    xtraj = np.concatenate([postraj, veltraj], axis=0)
    xtraj = PiecewisePolynomial.FirstOrderHold(time, xtraj)
    vis.visualize_trajectory(xtraj)

if __name__ == '__main__':
    reference = utils.load(REFSOURCE)
    reference['state'] = reference['_state']
    sim = utils.load(SIMSOURCE)
    # Check that the shapes match
    nT = min(reference['state'].shape[1], sim['state'].shape[1]) - 1
    reference['state'] = reference['state'][:, :nT]
    sim['state'] = sim['state'][:, :nT]
    sim['time'] = sim['time'][:nT]
    reference['time'] = reference['_time'][:nT]

    colors = [np.array([0.8, 0.8, 0.8, 0.4]), None]
    dataset = [reference, sim]
    make_a1_multivisualizer(dataset, colors)