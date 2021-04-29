"""
quaternion utilities

Luke Drnach
April 16, 2021
"""

import numpy as np

def quaternion_product(q1, q2):
    """
    Returns the quaternion product of two quaternions, q1*q2

    Arguments:
        q1: (4,) numpy array
        q2: (4,) numpy array
    """
    return np.hstack([
        q1[0]*q2[0] - np.dot(q1[1:], q2[1:]),
        q1[0] * q2[1:] + q2[0]*q1[1:] - np.cross(q1[1:], q2[1:])
    ])

def rpy_rates_to_velocity(rpy, rates):
    """
    Returns the angular velocity in body coordinates given roll-pitch-yaw euler angles and rates
    """
    return np.array([rates[0]*np.cos(rpy[1])*np.cos(rpy[2]) + rates[1]*np.sin(rpy[2]),
                    -rates[0]*np.sin(rpy[2])*np.cos(rpy[1]) + rates[1]*np.cos(rpy[2]),
                    rates[0]*np.sin(rpy[2]) + rates[2]])

def xyz_to_quaternion(xyz):
    """
        Converts XYZ-Fixed Angle Rotations to a quaternion

        Arguments:
            xyz: (3,) numpy array of x-y-z rotation angles
        Return values:
            (4,) numpy array, a unit quaternion
    """
    x, y, z = (xyz[0], xyz[1], xyz[2])
    qx = np.array([np.cos(x/2), np.sin(x/2), 0., 0.])
    qy = np.array([np.cos(y/2), 0., np.sin(y/2), 0.])
    qz = np.array([np.cos(z/2), 0., 0., np.sin(z/2)])
    return quaternion_product(qx, quaternion_product(qy, qz))

def quaternion_to_xyz(quat):
    """ 
        Convert a quaternion to X-Y-Z fixed angles
    """
    if quat.ndim == 1:
        quat = np.expand_dims(quat, axis=1)
    xyz = np.zeros((3, quat.shape[1]), dtype=quat.dtype)
    xyz[0,:] = np.arctan2(2*(quat[2,:]*quat[3,:] + quat[0,:]*quat[1,:]), quat[0,:]**2 + quat[1,:]**2 - quat[2,:]**2 - quat[3,:]**2)
    xyz[1,:] = -np.arcsin(2*(quat[3,:]*quat[1,:] - quat[0,:]* quat[2,:]))
    xyz[2,:] = np.arctan2(2*(quat[3,:]*quat[1,:] - quat[0,:]*quat[2,:]))
    if xyz.shape[1] == 1:
        return np.squeeze(xyz)
    else:
        return xyz

def quat2rpy(quat):
    """
    Convert a quaternion to Roll-Pitch-Yaw
    
    Arguments:
        quaternion: a (4,n) numpy array of quaternions
    
    Return values:
        rpy: a (3,n) numpy array of roll-pitch-yaw values
    """
    rpy = np.zeros((3, quat.shape[1]))
    rpy[0,:] = np.arctan2(2*(quat[0,:]*quat[1,:] + quat[2,:]*quat[3,:]),
                         1-2*(quat[1,:]**2 + quat[2,:]**2))
    rpy[1,:] = np.arcsin(2*(quat[0,:]*quat[2,:]-quat[3,:]*quat[1,:]))
    rpy[2,:] = np.arctan2(2*(quat[0,:]*quat[3,:]+quat[1,:]*quat[2,:]),
                        1-2*(quat[2,:]**2 + quat[3,:]**2))
    return rpy

def rpy_to_quaternion(rpy):
    """
    Convert roll-pitch-yaw angles to quaternions

    Arguments:
        rpy: a 3-array containing roll, pitch, and yaw angles in radians
    Return values:
        a 4-array containing the resulting quaterion in [w,x,y,z] convention, where w is the scalar part
    """
    r, p, y = (rpy[0], rpy[1], rpy[2])
    qx = np.array([np.cos(r/2), np.sin(r/2), 0, 0])
    qy = np.array([np.cos(p/2), 0, np.sin(p/2), 0])
    qz = np.array([np.cos(y/2), 0, 0, np.sin(y/2)])
    return quaternion_product(qz, quaternion_product(qy, qx))

def integrate_quaternion(q, w_axis, w_mag, dt):
    """
    Integrate the unit quaternion q given it's associated angular velocity w

    Arguments:
        q: (4,) numpy array specifying a unit quaternion
        w: (3,) numpy array specifying an angular velocity, as a unit vector
        w_mag:(1,) numpy array specifying the magnitude of the angular velocity (in world coordinates)
        dt: scalar indicating timestep 

    Return Values
        (4,) numpy array specifying the next step unit quaternion
    """
    # Exponential map of angular velocity
    delta_q = np.hstack([np.cos(w_mag * dt / 2.), w_axis * np.sin(w_mag * dt /2.)])
    # Do the rotation for body-fixed angular rate
    return quaternion_product(delta_q, q)

def integrate_quaternion_in_body(q, w_axis, w_mag, dt):
    """
    Integrate the unit quaternion q given it's associated angular velocity w

    Arguments:
        q: (4,) numpy array specifying a unit quaternion
        w: (3,) numpy array specifying an angular velocity, as a unit vector
        w_mag:(1,) numpy array specifying the magnitude of the angular velocity (in body coordinates)
        dt: scalar indicating timestep 

    Return Values
        (4,) numpy array specifying the next step unit quaternion
    """
    # Exponential map of angular velocity
    delta_q = np.hstack([np.cos(w_mag * dt / 2.), w_axis * np.sin(w_mag * dt /2.)])
    # Do the rotation for body-fixed angular rate
    return quaternion_product(q, delta_q)