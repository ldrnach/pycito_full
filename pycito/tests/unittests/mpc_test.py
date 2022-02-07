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
from pycito.systems.block.block import Block
#TODO: Check that the "getTime" for maximum time is actually correct
#TODO: Unittesting for MPC
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
        # Initialize the mpc class
        plant = A1VirtualBase()
        plant.Finalize()
        filename = os.path.join('pycito','tests','data','a1_step.pkl')
        lintraj = mpc.LinearizedContactTrajectory.load(plant, filename)
        cls.horizon = 1
        cls.mpc = mpc.LinearContactMPC(lintraj, horizon=cls.horizon)
        # Create an example mpc program
        cls.start_idx = int(lintraj.num_timesteps/2)
        t = lintraj._time[cls.start_idx]
        x = lintraj.getState(cls.start_idx)
        cls.mpc.create_mpc_program(t, x)

    def test_num_costs(self):
        """Check that MPC adds the correct number of cost terms"""
        cost_per_step = 5 + 4   # 5 Quadratic costs + 4 complementarity costs
        ncosts = len(self.mpc.prog.GetAllCosts())
        self.assertEqual(ncosts, self.horizon * cost_per_step, msg="Unexpected number of total cost terms")

    def test_num_constraints(self):
        """Check that MPC adds the expected number of constraint terms"""
        cstr_per_step = 1 + 2 + 2 + 2 + 2       # Dynamics (1), Distance (2), Dissipation (2), Friction (2), JointLimits (2)
        ncstr = len(self.mpc.prog.GetAllConstraints())
        self.assertEqual(ncstr, self.horizon*cstr_per_step + 1, msg="Unexpected number of total constraint terms")

    def test_num_decision_variables(self):
        """Check that MPC adds the expected number of decision variables"""
        nX, nU, nF, nS, nJ = self.mpc.state_dim, self.mpc.control_dim, self.mpc.force_dim, self.mpc.slack_dim, self.mpc.jlimit_dim
        nC = nF + nS + nJ # Number of complementarity variables
        nvars = (nX + nU + nF + nS + nJ + nC)*self.horizon + nX
        self.assertEqual(self.mpc.prog.num_vars(), nvars, msg='Unexpected number of decision variables')

    def test_solve(self):
        """
        Check that we can solve the MPC program successfully
        """
        result = self.mpc.solve()
        self.assertTrue(result.is_success(), msg=f"failed to solve MPC successfully using default solver {result.get_solver_id().name()}")
        self.check_solution(result)

    def check_solution(self, result):
        """
        Check the solution from MPC
        """
        states = result.GetSolution(self.mpc.dx)
        controls = result.GetSolution(self.mpc.du)
        forces = result.GetSolution(self.mpc.dl)
        velslack = result.GetSolution(self.mpc.ds)
        jointlimit = result.GetSolution(self.mpc.djl)
        # Assert that the correct values for states and controls are zero
        np.testing.assert_allclose(states, np.zeros((self.mpc.state_dim, self.horizon+1)), atol=1e-6, err_msg=f"MPC returned significantly nonzero state error")
        np.testing.assert_allclose(controls. np.zeros((self.mpc.control_dim, self.horizon)), atol=1e-6, err_msg=f"MPC returned significantly nonzero control error")
        # Assert that the correct values for forces and slacks are the trajectory values
        expected_force = np.concatenate([self.lintraj.GetForce(n) for n in range(self.start_idx, self.start_idx + self.horizon)], axis=1)
        expected_slack = np.concatenate([self.lintraj.GetSlack(n) for n in range(self.start_idx, self.start_idx + self.horizon)], axis=1)
        expected_jlim  = np.concatenate([self.lintraj.GetJointLimit(n) for n in range(self.start_idx, self.start_idx + self.horizon)], axis=1)
        np.testing.assert_allclose(forces, expected_force, atol=1e-6, err_msg="Forces are significantly different from trajectory values")
        np.testing.assert_allclose(velslack, expected_slack, atol=1e-6, err_msg="slacks are significantly different from trajectory values")
        np.testing.assert_allclose(jointlimit, expected_jlim, atol=1e-6, err_msg='Joint limits are significantly different from trajectory values')

class LinearTrajectoryVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        #TODO: Check that the dynamics are satisfied exactly
        cls.x = np.array([[ 0.0, 0.1],
                          [ 0.5, 0.5],
                          [ 0.0, 1.0],
                          [ 0.0, 0.0]
                        ])
        cls.time = np.array([0, 0.1])
        cls.h = np.array([0.1])
        cls.u = np.array([[0.0, 14.905]])      
        cls.l = np.array([[9.81, 9.81],
                          [0.0, 0.0],
                          [0.0, 0.0],
                          [0.0, 4.905], 
                          [0.0, 0.0],
                          [0.0, 1.0]
                         ]) 
        cls.plant = Block()
        cls.plant.Finalize()
        cls.lintraj = mpc.LinearizedContactTrajectory(cls.plant, cls.time, cls.x, cls.u, cls.l)

    def test_example_dynamics(self):
        """Assert that the example is correct as written"""
        # Check the dynamics
        dyn = cstr.BackwardEulerDynamicsConstraint(self.plant)
        out = dyn(np.concatenate([self.h, self.x[:, 0], self.x[:, 1], self.u[:, 1], self.l[:-1, 1]], axis=0))
        np.testing.assert_allclose(out, np.zeros((4,)), atol=1e-8, err_msg='Example dynamics not satisfied')
        # Check the normal distances
        dist = cstr.NormalDistanceConstraint(self.plant)
        np.testing.assert_allclose(dist(self.x[:, 0]), np.zeros((1,)), atol=1e-8, err_msg="Example normal distance not satisfied at index 0")
        np.testing.assert_allclose(dist(self.x[:, 1]), np.zeros((1,)), atol=1e-8, err_msg="Example normal distance not satisfied at index 0")
        # Check the dissipations
        diss = cstr.MaximumDissipationConstraint(self.plant)
        out0 = diss(np.concatenate([self.x[:, 0], self.l[-1:, 0]], axis=0))
        np.testing.assert_allclose(out0, np.zeros((4,)), atol=1e-8, err_msg="Dissipation constraint not satisfied at index 0")
        out1 = diss(np.concatenate([self.x[:, 1], self.l[-1:, 1]], axis=0))
        np.testing.assert_allclose(out1, np.array([2., 1., 0., 1.]), atol=1e-8, err_msg="Dissipation constraint not satisfied at index 1")
        # Check the friction cones
        fric = cstr.FrictionConeConstraint(self.plant)
        out0 = fric(np.concatenate([self.x[:, 0], self.l[:5, 0]], axis=0))
        np.testing.assert_allclose(out0, np.array([4.905]), atol=1e-8, err_msg="Friction cone constraint not satisfied at index 0")
        out1 = fric(np.concatenate([self.x[:, 1], self.l[:5, 1]], axis=0))
        np.testing.assert_allclose(out1, np.zeros((1, )), atol=1e-8, err_msg="Friction cone constriant not satisfied at index 1")

    def test_linear_dynamics(self):
        """Verify that the linearized dynamics parameters are accurate"""
        # Get the contact Jacobian
        context = self.plant.multibody.CreateDefaultContext()
        self.plant.multibody.SetPositionsAndVelocities(context, self.x[:, 1])
        Jn, Jt = self.plant.GetContactJacobians(context)
        J = np.column_stack((Jn.T, Jt.T))
        # Construct the linear dynamics matrix - it's constant, because the block system is linear
        Ax_0 = -np.eye(4)
        Ax_1 =  np.eye(4)
        Ax_1[:2, 2:] = -np.eye(2) * self.h
        Au = np.zeros((4,1))
        Au[2, 0] = -self.h  
        Al = np.vstack([np.zeros_like(J), -self.h*J])     
        A_expected = np.column_stack([Ax_0, Ax_1, Au, Al])
        b_expected = -Al.dot(self.l[:-1, 1])
        # Check the dynamics parameters
        np.testing.assert_allclose(self.lintraj.dynamics_cstr[0].A, A_expected, atol=1e-8, err_msg="Linear dynamics A matrix is inaccurate")
        np.testing.assert_allclose(self.lintraj.dynamics_cstr[0].b, b_expected, atol=1e-8, err_msg="Linear dynamics c vector is inaccurate")

    def test_linear_distance(self):
        """Verify that the normal distance parameters are accurate"""
        A_expected = np.array([[0, 1, 0, 0]])
        c_expected_0 = np.array([self.x[1, 0] - 0.5])
        c_expected_1 = np.array([self.x[1, 1] - 0.5])
        # Check the linearized distance parameters
        np.testing.assert_allclose(self.lintraj.distance_cstr[0].A, A_expected, atol=1e-8, err_msg="Linearized distance constraint A matrix is inaccurate at index 0")
        np.testing.assert_allclose(self.lintraj.distance_cstr[1].A, A_expected, atol=1e-8, err_msg="Linearized distance constraint A matrix is inaccurate at index 1")
        np.testing.assert_allclose(self.lintraj.distance_cstr[0].c, c_expected_0, atol=1e-8, err_msg="Linearized distance constraint c vector is inaccurate at index 0")
        np.testing.assert_allclose(self.lintraj.distance_cstr[1].c, c_expected_1, atol=1e-8, err_msg="Linearized distance constraint c vector is inaccurate at index 1")

    def test_linear_dissipation(self):
        """Verify that the linearized dissipation parameters are accurate"""
        # Collect the expected values
        context = self.plant.multibody.CreateDefaultContext()
        self.plant.multibody.SetPositionsAndVelocities(context, self.x[:,0])
        _, Jt = self.plant.GetContactJacobians(context)
        e = self.plant.duplicator_matrix()
        A_expected_0 = np.column_stack([np.zeros((4,2)), Jt, e.T])
        c_expected_0 = Jt.dot(self.x[2:, 0])
        # Second index
        self.plant.multibody.SetPositionsAndVelocities(context, self.x[:, 1])
        _, Jt = self.plant.GetContactJacobians(context)
        A_expected_1 = np.column_stack([np.zeros((4,2)), Jt, e.T])
        c_expected_1 = Jt.dot(self.x[2:, 1])
        # Check the linearized dissipation parameters
        np.testing.assert_allclose(self.lintraj.dissipation_cstr[0].A, A_expected_0, atol=1e-8, err_msg="Linearized dissipation constraint A matrix is inaccurate at index 0")
        np.testing.assert_allclose(self.lintraj.dissipation_cstr[1].A, A_expected_1, atol=1e-8, err_msg="Linearized dissipation constraint A matrix is inaccurate at index 1")
        np.testing.assert_allclose(self.lintraj.dissipation_cstr[0].c, c_expected_0, atol=1e-8, err_msg="Linearized dissipation constraint c vector is inaccurate at index 0")
        np.testing.assert_allclose(self.lintraj.dissipation_cstr[1].c, c_expected_1, atol=1e-8, err_msg="Linearized dissipation constraint c vector is inaccurate at index 1")
    
    def test_linear_friction(self):
        """Verify that the linear friction cone parameters are accurate"""
        # Collect the expected values
        mu = np.atleast_2d(self.plant.terrain.friction)
        e = self.plant.duplicator_matrix()
        A_expected = np.column_stack([np.zeros((1,4)), mu, -e])
        c_expected = np.zeros((1,))
        # Check the linearized friction cone parameters
        np.testing.assert_allclose(self.lintraj.friccone_cstr[0].A, A_expected, atol=1e-8, err_msg="Linearized friction cone constraint A matrix is inaccurate at index 0")
        np.testing.assert_allclose(self.lintraj.friccone_cstr[1].A, A_expected, atol=1e-8, err_msg="Linearized friction cone constraint A matrix is inaccurate at index 1")
        np.testing.assert_allclose(self.lintraj.friccone_cstr[0].c, c_expected, atol=1e-8, err_msg="Linearized friction cone constraint c vector is inaccurate at index 0")
        np.testing.assert_allclose(self.lintraj.friccone_cstr[1].c, c_expected, atol=1e-8, err_msg="Linearized friction cone constraint c vector is inaccurate at index 1")


if __name__ == '__main__':
    unittest.main()