"""
unittests for contact-implicit model predictive controller

Luke Drnach
February 2, 2022
"""
import numpy as np
import os, unittest

from pycito.controller import mpc, mlcp
from pycito.trajopt import constraints as cstr
from pycito.systems.A1.a1 import A1VirtualBase
#TODO: Check that the "getTime" for maximum time is actually correct
#TODO: Unittesting for LinearContactTrajectory, MPC
class ReferenceTrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plant = A1VirtualBase()
        plant.Finalize()
        filename = os.path.join("pycito","tests","data","a1_step.pkl")
        cls.reftraj = mpc.ReferenceTrajectory.load(plant, filename)

    def test_dims(self):
        """
        Check all dimensions of the trajectories stored in ReferenceTrajectory
        """
        # shorthand parameters
        nX = self.reftraj.plant.multibody.num_positions() + self.reftraj.plant.multibody.num_velocities()
        nU = self.reftraj.plant.multibody.num_actuators()
        nC = self.reftraj.plant.num_contacts()
        nF = self.reftraj.plant.num_friction()
        nJ = np.sum(np.isfinite(self.reftraj.plant.get_joint_limits()))
        # Check the dimensions
        self.assertEqual(self.reftraj.state_dim, nX, f"Expected {nX} states, got {self.reftraj.state_dim} instead")
        self.assertEqual(self.reftraj.control_dim, nU, f"Expected {nU} control variables, got {self.reftraj.control_dim} instead")
        self.assertEqual(self.reftraj.force_dim, nC+nF, f"Expected {nC+nF} force variables, got {self.reftraj.force_dim} instead")
        self.assertEqual(self.reftraj.slack_dim, nC, f"Expected {nC} slack variables, got {self.reftraj.slack_dim} instead")
        self.assertEqual(self.reftraj.jlimit_dim, nJ, f"Expected {nJ} joint limit variables, got {self.reftraj.jlimit_dim} instead")
        self.assertEqual(self.reftraj.num_timesteps, self.reftraj._time.shape[0], f"Unexpected number of time samples")

    def test_has_joint_limits(self):
        """Check that the has_joint_limits property works correctly"""
        self.assertTrue(self.reftraj.has_joint_limits, f"has_joint_limits returns a non-True value")

    def test_get_time(self):
        """
        Check the 'getTimeIndex' method
        """
        # Test value is in array
        idx = 10
        t_test = self.reftraj._time[idx]
        self.assertEqual(self.reftraj.getTimeIndex(t_test), idx, f"getTimeIndex failed to return correct index for a time value within the time array")
        # Test value is not in array, but within bounds
        t_test = (self.reftraj._time[idx] + self.reftraj._time[idx+1])/2
        self.assertEqual(self.reftraj.getTimeIndex(t_test), idx, f"getTimeIndex failed to return correct index for time within expected bounds")
        # Test value is less than the minimum time
        t_test = self.reftraj._time[0] - 1.0
        self.assertEqual(self.reftraj.getTimeIndex(t_test), 0, f"getTimeIndex failed to return 0 when test time was less than minimum time")
        # Test value is greater than the maximum time
        t_test = self.reftraj._time[-1] + 1.0
        self.assertEqual(self.reftraj.getTimeIndex(t_test), self.reftraj.num_timesteps, f"getTimeIndex failed to return maximum index when test time is greater than maximum time")

    def test_get_state(self):
        """
        Test that we can get the state, with wraparound
        """
        N = self.reftraj.num_timesteps
        x0 = self.reftraj.getState(0)
        xF = self.reftraj.getState(N)
        xmid = self.reftraj.getState(int(N/2))
        # Test that the states returned are different
        self.assertFalse(np.allclose(x0, xmid),"getState returns initial state for a nonzero index")
        self.assertFalse(np.allclose(xmid, xF),"getState returns final state for an intermediate index") 
        # Test the index is less than 0
        np.testing.assert_allclose(self.reftraj.getState(-1), x0, atol=1e-7, err_msg="getState does not return the initial state when the index is less than 0")
        # Test the index is greater than the maximum
        np.testing.assert_allclose(self.reftraj.getState(N+1), xF, atol=1e-7, err_msg="getState does not return the final state when the index exceeds maximum")

    def test_get_control(self):
        """
        Test we can get the control, with wraparound
        """
        N = self.reftraj.num_timesteps
        u0 = self.reftraj.getControl(0)
        uN = self.reftraj.getControl(N)
        umid = self.reftraj.getControl(int(N/2))
        # Test that the controls are different
        self.assertFalse(np.allclose(u0, umid), "getControl returns initial control for nonzero index")
        self.assertFalse(np.allclose(umid, uN), "getControl returns final control for an intermediate index")
        # Test when index is less than 0
        np.testing.assert_allclose(self.reftraj.getControl(-1), u0, atol=1e-6, err_msg="getControl does not return the initial control when index is less than 0")
        # Test when index is greater than max
        np.testing.assert_allclose(self.reftraj.getControl(N+1), uN, atol=1e-6, err_msg="getControl does not return the final state when teh index exceeds maximum")

    def test_get_force(self):
        """
        test we can get the force, with wraparound
        """
        N = self.reftraj.num_timesteps
        f0 = self.reftraj.getForce(0)
        fN = self.reftraj.getForce(N)
        fmid = self.reftraj.getForce(int(N/2))
        # Test that the forces are different
        self.assertFalse(np.allclose(f0, fmid), "getForce returns initial force for a nonzero index")
        self.assertFalse(np.allclose(fmid, fN), "getForce returns final force for an intermediate index")
        # Test when index is less than 0
        np.testing.assert_allclose(self.reftraj.getForce(-1), f0, atol=1e-6, err_msg="getForce does not return the initial forces for index less than 0")
        np.testing.assert_allclose(self.reftraj.getForce(N+1), fN, atol=1e-6, err_msg="getForce does not return the final forces for index greater than maximum")

    def test_get_slack(self):
        """
        test we can get the slack, with wraparound
        """
        N = self.reftraj.num_timesteps
        s0 = self.reftraj.getSlack(0)
        sN = self.reftraj.getSlack(N)
        smid = self.reftraj.getSlack(int(N/2))
        # Test that the forces are different
        self.assertFalse(np.allclose(s0, smid), "getSlack returns initial slack for a nonzero index")
        self.assertFalse(np.allclose(smid, sN), "getSlack returns final slack for an intermediate index")
        # Test when index is less than 0
        np.testing.assert_allclose(self.reftraj.getSlack(-1), s0, atol=1e-6, err_msg="getSlack does not return the initial slack for index less than 0")
        np.testing.assert_allclose(self.reftraj.getSlack(N+1), sN, atol=1e-6, err_msg="getSlack does not return the final slack for index greater than maximum")

    def test_get_joint_limit(self):
        """
        test we can get the force, with wraparound
        """
        #NOTE: Joint limits are usually zero, so it's not effective to test intermediate points
        N = self.reftraj.num_timesteps
        jl0 = self.reftraj.getJointLimit(0)
        jlN = self.reftraj.getJointLimit(N)
        # Test when index is less than 0
        np.testing.assert_allclose(self.reftraj.getJointLimit(-1), jl0, atol=1e-6, err_msg="getJointLimit does not return the initial forces for index less than 0")
        # Test when index is greater than maximum
        np.testing.assert_allclose(self.reftraj.getJointLimit(N+1), jlN, atol=1e-6, err_msg="getJointLimit does not return the final forces for index greater than maximum")

class LinearContactTrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load all the trajectories 
        plant = A1VirtualBase()
        plant.Finalize()
        filename = os.path.join("pycito",'tests','data','a1_step.pkl')
        cls.lcptype = mlcp.CostRelaxedMixedLinearComplementarity
        cls.lintraj = mpc.LinearizedContactTrajectory.load(plant, filename, cls.lcptype)
    
    def test_get_dynamics(self):
        """
        Check that there are the expected number of dynamics linearization constraint objects
        Check that we can get a linearized dynamics constraint object
        Check that getting the constraints obeys index clipping
        """
        # Check the expected number of linearized dynamics constraints
        N = self.lintraj.num_timesteps
        self.assertEqual(len(self.lintraj.dynamics_cstr), N-1, msg=f"Unexpected number of linearized dynamics constraints")
        # Check the linearized dynamics object
        self.assertTrue(isinstance(self.lintraj.getDynamicsConstraint(int(N/2)), cstr.LinearImplicitDynamics), msg='getDynamicsConstraint does not return the correct type')
        # Check for index clipping
        self.assertEqual(self.lintraj.getDynamicsConstraint(N+1), self.lintraj.dynamics_cstr[-1], msg="getDynamicsConstraint does not return final constraint for indices greater than maximum")
        self.assertEqual(self.lintraj.getDynamicsConstraint(-1), self.lintraj.dynamics_cstr[0], msg="getDynamicsConstraint does not return initial constraint for indices less than 0")

    def test_get_distance(self):
        """
        Check that there are the expected number of linearized distance constraint objects
        Check that we can get a linearized distance constraint object
        Check that getting the constraints obeys index clipping
        """
        # Check the expected number of linearized distance constraints
        N = self.lintraj.num_timesteps
        self.assertEqual(len(self.lintraj.distance_cstr), N, msg=f"Unexpected number of linearized distance constraints")
        # Check the linearized dynamics object
        self.assertTrue(isinstance(self.lintraj.getDistanceConstraint(int(N/2)), self.lcptype), msg='getDistanceConstraint does not return the correct type')
        # Check for index clipping
        self.assertEqual(self.lintraj.getDistanceConstraint(N+1), self.lintraj.distance_cstr[-1], msg="getDistanceConstraint does not return final constraint for indices greater than maximum")
        self.assertEqual(self.lintraj.getDistanceConstraint(-1), self.lintraj.distance_cstr[0], msg="getDistanceConstraint does not return initial constraint for indices less than 0")

    def test_get_dissipation(self):
        """
        Check that there are the expected number of linearized dissipation constraint objects
        Check that we can get a linearized dissipation constraint object
        Check that getting the constraints obeys index clipping
        """
        # Check the expected number of linearized dynamics constraints
        N = self.lintraj.num_timesteps
        self.assertEqual(len(self.lintraj.dissipation_cstr), N, msg=f"Unexpected number of linearized dissipation constraints")
        # Check the linearized dynamics object
        self.assertTrue(isinstance(self.lintraj.getDissipationConstraint(int(N/2)), self.lcptype), msg='getDissipationConstraint does not return the correct type')
        # Check for index clipping
        self.assertEqual(self.lintraj.getDissipationConstraint(N+1), self.lintraj.dissipation_cstr[-1], msg="getDissipationConstraint does not return final constraint for indices greater than maximum")
        self.assertEqual(self.lintraj.getDissipationConstraint(-1), self.lintraj.dissipation_cstr[0], msg="getDissipationConstraint does not return initial constraint for indices less than 0")

    def test_get_friction_cone(self):
        """
        Check that there are the expected number of linearized friction cone constraint objects
        Check that we can get a linearized friction cone constraint object
        Check that getting the constraints obeys index clipping
        """
        # Check the expected number of linearized friction cone constraints
        N = self.lintraj.num_timesteps
        self.assertEqual(len(self.lintraj.friccone_cstr), N, msg=f"Unexpected number of linearized friction cone constraints")
        # Check the linearized dynamics object
        self.assertTrue(isinstance(self.lintraj.getFrictionConeConstraint(int(N/2)), self.lcptype), msg='getFrictionConeConstraint does not return the correct type')
        # Check for index clipping
        self.assertEqual(self.lintraj.getFrictionConeConstraint(N+1), self.lintraj.friccone_cstr[-1], msg="getFrictionConeConstraint does not return final constraint for indices greater than maximum")
        self.assertEqual(self.lintraj.getFrictionConeConstraint(-1), self.lintraj.friccone_cstr[0], msg="getFrictionConeConstraint does not return initial constraint for indices less than 0")

    def test_get_jointlimits(self):
        """
        Check that there are the expected number of dynamics linearization constraint objects
        Check that we can get a linearized dynamics constraint object
        Check that getting the constraints obeys index clipping
        """
        # Check the expected number of linearized dynamics constraints
        N = self.lintraj.num_timesteps
        self.assertEqual(len(self.lintraj.joint_limit_cstr), N, msg=f"Unexpected number of linearized joint limit constraints")
        # Check the linearized dynamics object
        self.assertTrue(isinstance(self.lintraj.getJointLimitConstraint(int(N/2)), self.lcptype), msg='getJointLimitConstraint does not return the correct type')
        # Check for index clipping
        self.assertEqual(self.lintraj.getJointLimitConstraint(N+1), self.lintraj.joint_limit_cstr[-1], msg="getJointLimitConstraint does not return final constraint for indices greater than maximum")
        self.assertEqual(self.lintraj.getJointLimitConstraint(-1), self.lintraj.joint_limit_cstr[0], msg="getJointLimitConstraint does not return initial constraint for indices less than 0")

class LinearContactMPCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

if __name__ == '__main__':
    unittest.main()