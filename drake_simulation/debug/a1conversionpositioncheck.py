import os
import numpy as np
from pycito.systems.visualization import Visualizer
import pycito.utilities as utils
from pydrake.all import (
    Quaternion, RollPitchYaw, PiecewisePolynomial,
    SpatialInertia, UnitInertia, PrismaticJoint, BallRpyJoint
)

#TODO: Check the floating base velocities

a1_urdf = os.path.join("systems","A1","A1_description","urdf","a1_no_collision.urdf")
a1_urdf = utils.FindResource(a1_urdf)

def make_virtual_joints(plant, model_index):
    # Add virtual joints to represent the floating base
    zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
    # Create virtual, zero-mass
    xlink = plant.AddRigidBody('xlink', model_index, zeroinertia)
    ylink = plant.AddRigidBody('ylink', model_index, zeroinertia)
    zlink = plant.AddRigidBody('zlink', model_index, zeroinertia)
    # Create the translational and rotational joints
    xtrans = PrismaticJoint("xtranslation", plant.world_frame(), xlink.body_frame(), [1., 0., 0.])
    ytrans = PrismaticJoint("ytranslation", xlink.body_frame(), ylink.body_frame(), [0., 1., 0.])
    ztrans = PrismaticJoint("ztranslation", ylink.body_frame(), zlink.body_frame(), [0., 0., 1.])
    rpyrotation = BallRpyJoint("baseorientation", zlink.body_frame(), plant.GetBodyByName('base', model_index).body_frame())
    # Add the joints to the multibody plant
    plant.AddJoint(xtrans)
    plant.AddJoint(ytrans)
    plant.AddJoint(ztrans)
    plant.AddJoint(rpyrotation)
    return plant

def quat2rpy(quat):
    rpy = []
    for q in quat.T:
        q = q / np.sqrt(np.sum(q ** 2))
        rpy.append(RollPitchYaw(Quaternion(q)).vector())
    return np.column_stack(rpy)


def quaternion_config_to_rpy_config(q):
    quat, pos, joint = q[:4, :], q[4:7,:], q[7:,:]
    rpy = quat2rpy(quat)
    q_rpy = np.row_stack([pos, rpy, joint])
    return q_rpy

def quaternion_vector(quaternion):
    return np.array([quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()])

def generate_test_trajectory(nQ):
    N = 100
    maxtheta = np.pi/3
    thetas = np.linspace(0, np.pi/3, N)
    q = np.zeros((nQ, N))
    # Orientation
    for k, theta in enumerate(thetas):
        rpy = RollPitchYaw(roll = theta, pitch = 0, yaw = 0)
        q[:4,k] = quaternion_vector(rpy.ToQuaternion())
    q2 = np.zeros((nQ, N))
    for k, theta in enumerate(thetas):
        rpy = RollPitchYaw(roll = maxtheta, pitch = theta, yaw = 0)
        q2[:4, k] = quaternion_vector(rpy.ToQuaternion())
    q3 = np.zeros((nQ, N))
    for k, theta  in enumerate(thetas):
        rpy = RollPitchYaw(roll = maxtheta, pitch= maxtheta , yaw = theta)
        q3[:4,k] = quaternion_vector(rpy.ToQuaternion())
    #COM Position shifts
    p = np.linspace(0, 1, N)
    q_com_x = np.zeros((nQ, N))
    q_com_x[4,:] = p
    q_com_y = np.zeros((nQ,N))
    q_com_y[4,:] = 1
    q_com_y[5,:] = p
    q_com_z = np.zeros((nQ, N))
    q_com_z[4,:] = 1
    q_com_z[5,:] = 1
    q_com_z[6,:] = p
    q_com = np.column_stack([q_com_x, q_com_y, q_com_z])
    q_com[:4, :] = q3[:4, -1:]
    # Joint position shifts
    q_joints = []
    theta = np.linspace(0, np.pi/2, N)
    for k in range(7,nQ):
        q_joint = np.zeros((nQ, N))
        q_joint[k,:] = theta
        q_joint[:7,:] = q_com[:7, -1:]
        q_joints.append(q_joint)
    q_joint = np.column_stack(q_joints)

    return np.column_stack([q, q2, q3, q_com, q_joint])


# Create the visualizer
vis = Visualizer(a1_urdf)
vis.addModelFromFile(a1_urdf, name=f"A1_RPY")
vis.plant = make_virtual_joints(vis.plant, vis.model_index[-1])
# Change the color of quaternion plant to white
for body_ind in vis.plant.GetBodyIndices(vis.model_index[0]):
    vis.setBodyColor(body_ind, np.array([0.8, 0.8, 0.8, 0.4]))
vis._finalize_plant()
# Create the trajectories
nT = 100
traj = generate_test_trajectory(vis.plant.num_positions(vis.model_index[0]))
# Convert from the Quaternion form to RollPitchYaw
converted_traj = quaternion_config_to_rpy_config(traj)
time = np.squeeze(np.linspace(0, 5, traj.shape[1]))
# Combine and visualize the trajectories
nv = vis.plant.num_velocities(vis.model_index[0])
nv2 = vis.plant.num_velocities(vis.model_index[1])
vel = np.zeros((nv+nv2, time.size))
total_traj = np.row_stack([converted_traj,traj, vel])
xtraj = PiecewisePolynomial.FirstOrderHold(time, total_traj)
# VISUALIZE!
vis.visualize_trajectory(xtraj)



