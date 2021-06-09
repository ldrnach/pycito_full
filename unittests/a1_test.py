"""
unittests for contactimplicit.py

Luke Drnach
October 14, 2020
"""

import numpy as np
import unittest
from systems.A1.a1 import A1
from pydrake.all import RigidTransform

class A1Test(unittest.TestCase):
    def test_make_a1_with_rpy(self):
        """ Test creating A1 from scratch with a roll-pitch-yaw joint"""
        a1_quaternion = A1()
        a1_quaternion.Finalize()
        a1_rpy = A1(urdf_file="systems/A1/A1_description/urdf/a1_foot_collision_no_floating.urdf")
        a1_rpy.multibody.WeldFrames(a1_rpy.multibody.world_frame(), a1_rpy.multibody.GetBodyByName("weld").body_frame(), RigidTransform())
        a1_rpy.Finalize()
        # Check the number of position and velocity variables
        rpy_pos = a1_rpy.multibody.num_positions()
        quat_pos = a1_quaternion.multibody.num_positions()
        self.assertEqual(rpy_pos, quat_pos-1, msg=f"Expected {quat_pos-1} positions, got {rpy_pos} instead")
        rpy_vel = a1_rpy.multibody.num_velocities()
        self.assertEqual(rpy_vel, rpy_pos, msg=f"Expected {rpy_pos} velocities, got {rpy_vel} instead")

    def test_floating_rpy_joint(self):
        """Verify that using the floating rpy joint removes a position variables from the generalized positions"""
        # Create an A1 model with a floating base using quaternions
        a1_quaternion = A1()
        a1_quaternion.Finalize()
        # Create an A1 model with a floating base using RPY joints
        a1_rpy = A1()
        a1_rpy.useFloatingRPYJoint()
        a1_rpy.Finalize()
        # Check that the RPY joint has 1 fewer position variables, but the same number of velocity variables
        quat_pos = a1_quaternion.multibody.num_positions()
        rpy_pos = a1_rpy.multibody.num_positions()
        self.assertEqual(rpy_pos, quat_pos-1, msg=f"FloatingRPYJoint has {rpy_pos} positions, expected {quat_pos-1}")
        rpy_vel = a1_rpy.multibody.num_velocities()
        self.assertEqual(rpy_vel, rpy_pos, msg=f"Expected {rpy_pos} generalized velocities for RPY floating joint, got {rpy_vel} instead")
