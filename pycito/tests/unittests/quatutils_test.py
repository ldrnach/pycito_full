"""
Unit tests for the quaternion utilities file, quatutils.py

Luke Drnach
April 29, 2021
"""

import numpy as np
import unittest
from pycito.trajopt.quatutils import *

class QuatUtilsTest(unittest.TestCase):
    def setUp(self):
        """
            Set values to be used in the tests
        """
        # XYZ-Fixed angles
        self.xyz = np.array([np.pi/2, np.pi/3, 2*np.pi/3])
        self.quat_xyz = np.array([np.sqrt(6.)/4., 0., np.sqrt(2.)/2., np.sqrt(2)/4])
        # XYZ-Euler angles (Roll-Pitch-Yaw)
        self.rpy = np.array([np.pi/2., np.pi/3., 2*np.pi/3.])
        self.quat_rpy = np.array([0., np.sqrt(6.)/4., -np.sqrt(2.)/4., np.sqrt(2.)/2.])
        # Quaternion multiplication
        self.q1 = np.array([1.1, 2.1, -4.0, 5.7])
        self.q2 = np.array([-2.3, 1.0, 3.5, 4.4])
        self.q1q2 = np.array([-15.71, -41.28, 9.51, 3.08])
        # Quaternion Integration
        self.w_axis = np.sqrt(3)*np.array([1., 1., 1])
        self.w_mag = np.array([2.5])
        self.dt = 0.1
        c = np.cos(0.125)
        s = np.sqrt(3)*np.sin(0.125)
        self.q_world = np.array([1.1*c - 3.8*s,
                                2.1*c + 10.8*s,
                                -4.0*c - 2.5*s, 
                                5.7*c - 5*s])        
        self.q_body = np.array([1.1*c - 3.8*s,
                                2.1*c - 8.6*s,
                                -4.0*c + 4.7*s,
                                5.7*c + 7.2*s])


    def test_quaternion_product(self):
        """ Test multiplication of two quaternions """
        # Test trivial case
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(quaternion_product(q0, self.q1), self.q1, err_msg="Quaternion product fails at identity")
        # Test non-trivial case
        qprod = quaternion_product(self.q1, self.q2)
        np.testing.assert_allclose(qprod, self.q1q2, err_msg="Quaternion multiplication fails on non-trivial case")

    def test_xyz_to_quaternion(self):
        """ Test conversion of xyz-fixed angles to quaternion """
        # First test the trivial case
        q = xyz_to_quaternion(np.zeros(3,))
        np.testing.assert_allclose(q, np.array([1, 0, 0, 0]), err_msg="XYZ-to-quaternion fails at origin")
        # Test the elementary rotations
        angle = np.pi/3.
        # X-Axis only rotation
        quat = xyz_to_quaternion(np.array([angle, 0., 0.]))
        np.testing.assert_allclose(quat, np.array([np.cos(angle/2), np.sin(angle/2), 0., 0.]), atol=1e-12, rtol=1e-7, err_msg="XYZ-to-Quaternion fails on X-only")
        # Y-Axis only rotation
        quat = xyz_to_quaternion(np.array([0., angle, 0.]))
        np.testing.assert_allclose(quat, np.array([np.cos(angle/2), 0., np.sin(angle/2), 0.]), atol=1e-12, rtol=1e-7, err_msg='XYZ-to-Quaternion fails on Y-only rotation')
        # Z-axis only rotation
        quat = xyz_to_quaternion(np.array([0., 0., angle]))
        np.testing.assert_allclose(quat, np.array([np.cos(angle/2), 0., 0., np.sin(angle/2)]), atol=1e-12, rtol=1e-7, err_msg="XYZ-to-Quaternion fails on Z-only rotation")
        # Now test a non-trivial case
        q = xyz_to_quaternion(self.xyz)
        np.testing.assert_allclose(q, self.quat_xyz, atol=1e-12, rtol=1e-7, err_msg="XYZ-to-quaternion fails in non-trivial case")        
        
    def test_quaternion_to_xyz(self):
        """ Test conversion of quaternion to xyz-fixed angles """
        # Test the trivial case at origin first
        xyz = quaternion_to_xyz(np.array([1., 0., 0., 0.]))
        np.testing.assert_allclose(xyz, np.zeros((3,)), atol=1e-12, rtol=1e-7, err_msg="Quat-to-XYZ fails at origin")
        # Test the elementary cases - X-Rotation only
        angle = np.pi/3.
        xyz = quaternion_to_xyz(np.array([np.cos(angle/2), np.sin(angle/2), 0., 0.]))
        np.testing.assert_allclose(xyz, np.array([angle, 0., 0.]), atol=1e-12, rtol=1e-7, err_msg='Quaternion-to-XYZ fails for X-only rotation')
        # Test the elementary cases - Y-Rotation only
        xyz = quaternion_to_xyz(np.array([np.cos(angle/2), 0., np.sin(angle/2), 0., 0.]))
        np.testing.assert_allclose(xyz, np.array([0., angle, 0.]), atol=1e-12, rtol=1e-7, err_msg="Quaternion-to-XYZ fails on Y-only rotation")
        # Test the elementary cases - Z-Rotation only
        xyz = quaternion_to_xyz(np.array([np.cos(angle/2), 0., 0., np.sin(angle/2)]))
        np.testing.assert_allclose(xyz, np.array([0., 0., angle]), atol=1e-12, rtol=1e-7, err_msg='Quaternion-to-XYZ fails on Z-only rotation')
        # Test in non-trivial case
        xyz = quaternion_to_xyz(self.quat_xyz)
        np.testing.assert_allclose(xyz, self.xyz, atol=1e-12, rtol=1e-7, err_msg="Quat-to-XYZ fails in non-trivial case")

    def test_rpy_to_quaternion(self):
        """ Test conversion of roll-pitch-yaw to quaternion """
        # Test in the trivial case at origin first
        quat = rpy_to_quaternion(np.zeros((3,)))
        np.testing.assert_allclose(quat, np.array([1., 0., 0., 0.]), atol=1e-12, rtol=1e-7, err_msg="RPY-to-Quaternion fails at origin")
        # Test the elementary rotations - Roll only (X)
        angle = np.pi/3
        quat_x = rpy_to_quaternion(np.array([angle, 0., 0.]))
        np.testing.assert_allclose(quat_x, np.array([np.cos(angle/2), np.sin(angle/2), 0., 0.]), atol=1e-12, rtol=1e-7, err_msg="RPY-To-Quaternion fails on X-axis rotation")
        # Test the elementary rotations - pitch (Y) only
        quat_y = rpy_to_quaternion(np.array([0., angle, 0.]))
        np.testing.assert_allclose(quat_y, np.array([np.cos(angle/2), 0., np.sin(angle/2), 0.]), atol=1e-12, rtol=1e-7, err_msg="RPY-to-Quaternion fails on Y-Axis rotation")
        # Test the elementary rotations - yaw (Z) only
        quat_z = rpy_to_quaternion(np.array([0., 0., angle]))
        np.testing.assert_allclose(quat_z, np.array([np.cos(angle/2), 0., 0. ,np.sin(angle/2)]), atol=1e-12, rtol=1e-7, err_msg="RPY-to-Quaternion fails on Z-Axis rotation")        
        # Test in non-trivial case
        quat = rpy_to_quaternion(self.rpy)
        np.testing.assert_allclose(quat, self.quat_rpy, atol=1e-12, rtol=1e-7, err_msg="RPY-to-Quaternion fails at non-trivial case")

    def test_quaternion_to_rpy(self):
        """ Test conversion of quaternion to roll-pitch-yaw angles"""
        # Test the trivial case
        rpy = quaternion_to_rpy(np.array([1., 0., 0., 0.]))
        np.testing.assert_allclose(rpy, np.zeros((3,)), err_msg="Quaternion-to-RPY fails at origin")
        # Test the elementary rotations - Roll (X) only
        angle = np.pi/3
        rpy = quaternion_to_rpy(np.array([np.cos(angle/2), np.sin(angle/2), 0., 0.]))
        np.testing.assert_allclose(rpy, np.array([angle, 0., 0.]), err_msg="Quaternion-to-RPY fails on Roll-only rotation")
        # Test elementary rotation - Pitch (Y) only
        rpy = quaternion_to_rpy(np.array([np.cos(angle/2), 0., np.sin(angle/2), 0.]))
        np.testing.assert_allclose(rpy, np.array([0., angle, 0.]), err_msg="Quaternion-to-RPY fails on Pitch-only rotation")
        # Test the yaw-only elementary rotation
        rpy = quaternion_to_rpy(np.array([np.cos(angle/2), 0., 0., np.sin(angle/2)]))
        np.testing.assert_allclose(rpy, np.array([0., 0., angle]), err_msg="Quaternion-to-RPY fails on Yaw-only rotation")
        # Test in the non-trivial case
        rpy = quaternion_to_rpy(self.quat_rpy)
        np.testing.assert_allclose(rpy, self.rpy, err_msg="Quaternion-to-RPY fails at non-trivial case")

    def test_integrate_quaternion(self):
        """ Test integration of quaternion """
        # Test trivial integration at the origin (arbitrary axis)
        q_int = integrate_quaternion(self.q1, w_axis=np.ones((3,)), w_mag=np.zeros((1,)), dt=0.01)
        np.testing.assert_allclose(q_int, self.q1, err_msg="Quaternion integration fails when velocity is 0")
        # Test trivial integration at the origin (zero axis)
        q_int = integrate_quaternion(self.q1, w_axis=np.zeros((3,)), w_mag=np.zeros((1,)), dt=0.01)
        np.testing.assert_allclose(q_int, self.q1, err_msg="Quaternion itegration fails when the axis is zero")
        # Test non-trivial integration
        q_int = integrate_quaternion(self.q1, self.w_axis, self.w_mag, self.dt)
        np.testing.assert_allclose(q_int, self.q_world, err_msg="Quaternion integration fails in non-trivial case")

    def test_integrate_quaternion_body(self):
        """ Test integration of quaternion using body-fixed velocities"""
        # Test trivial integration at the origin (arbitrary axis)
        q_int = integrate_quaternion_in_body(self.q1, w_axis=np.ones((3,)), w_mag=np.zeros((1,)), dt=0.01)
        np.testing.assert_allclose(q_int, self.q1, err_msg="Quaternion integration fails when velocity is 0")
        # Test trivial integration at the origin (zero axis)
        q_int = integrate_quaternion_in_body(self.q1, w_axis=np.zeros((3,)), w_mag=np.zeros((1,)), dt=0.01)
        np.testing.assert_allclose(q_int, self.q1, err_msg="Quaternion itegration fails when the axis is zero")
        # Test non-trivial integration
        q_int = integrate_quaternion_in_body(self.q1, self.w_axis, self.w_mag, self.dt)
        np.testing.assert_allclose(q_int, self.q_body, err_msg="Quaternion integration in body coordinates fails in non-trivial case")

if __name__ == '__main__':
    unittest.main()