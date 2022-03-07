"""
unittests for pycito.controller.contactestimator.py

Luke Drnach
March 1, 2022
"""
import numpy as np
import unittest

from pydrake.all import MathematicalProgram, Solve

import pycito.controller.contactestimator as ce 
from pycito.systems.contactmodel import SemiparametricContactModel
from pycito.systems.block.block import Block
class SemiparametricFrictionConstraintTest(unittest.TestCase):
    def setUp(self):
        mu = np.array([1, .1])
        self.fN_desired = np.array([10, 15])
        kernel = np.diag([1, 3])
        self.fT_desired = np.array([4, 1, 3, 3])      
        self.vS_desired = np.array([0, 0])
        self.w_desired = np.array([-0.5, 0.1])   
        duplicator = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        # Create the constraint
        self.cstr = ce.SemiparametricFrictionConeConstraint(mu, kernel, duplicator)
        # Create a mathematical program
        self.prog = MathematicalProgram()
        self.fN = self.prog.NewContinuousVariables(rows=2, name='normal_forces')
        self.fT = self.prog.NewContinuousVariables(rows=4, name='friction_forces')
        self.vS = self.prog.NewContinuousVariables(rows=2, name='velocity_slack')
        self.w = self.prog.NewContinuousVariables(rows=2, name='weights')
        # Add the constraint
        self.cstr.addToProgram(self.prog, np.hstack([self.w, self.fN, self.fT]), self.vS)
        # Add error costs to regularize the program
        self.prog.AddQuadraticErrorCost(np.eye(2), self.fN_desired, self.fN)
        self.prog.AddQuadraticErrorCost(np.eye(4), self.fT_desired, self.fT)
        self.prog.AddQuadraticErrorCost(np.eye(2), self.vS_desired, self.vS)
        # Set initial guesses
        self.prog.SetInitialGuess(self.fN, self.fN_desired)
        self.prog.SetInitialGuess(self.fT, self.fT_desired)
        self.prog.SetInitialGuess(self.vS, self.vS_desired)
        self.prog.SetInitialGuess(self.w, self.w_desired)        
        self.prog.SetInitialGuess(self.cstr.relax, np.zeros((1,)))

    def test_num_constraints(self):
        """Check that we've added the expected number of constraints"""
        self.assertEqual(len(self.prog.GetAllConstraints()), 2, msg='Unexpected number of constraints')

    def test_num_costs(self):
        """Check that we've added the expected number of costs"""
        self.assertEqual(len(self.prog.GetAllCosts()), 4, msg='Unexpected number of cost terms')

    def test_eval_fric_coeff(self):
        """Test evaluation of the fricition coefficient"""
        mu = self.cstr.eval_friction_coeff(self.w_desired)
        np.testing.assert_allclose(mu, np.array([0.5, 0.4]), atol=1e-6, err_msg=f"Unexpected friction coefficients")

    def test_eval_friction_cone(self):
        """Test evaluation of the friction cone constraint"""
        dvals = np.concatenate([self.w_desired, self.fN_desired, self.fT_desired])
        fc = self.cstr.eval_frictioncone(dvals)
        np.testing.assert_allclose(fc, np.zeros((2,)), atol=1e-6, err_msg=f"Unexpected result evaluating friction cone")

    def test_solve(self):
        """Check that the program can be solved (even with the initial guess as a solution)"""
        result = Solve(self.prog)
        self.assertTrue(result.is_success(), f"Failed to solve with solver {result.get_solver_id().name()}")
        # Check the solution
        np.testing.assert_allclose(result.GetSolution(self.fN), self.fN_desired, atol=1e-6, err_msg=f'Unexpected solution for normal forces')
        np.testing.assert_allclose(result.GetSolution(self.fT), self.fT_desired, atol=1e-6, err_msg=f"Unexpected solution for friction forces")
        np.testing.assert_allclose(result.GetSolution(self.vS), self.vS_desired, atol=1e-6, err_msg=f"Unexpecetd solution for the velocity slacks")
        np.testing.assert_allclose(result.GetSolution(self.w), self.w_desired, atol=1e-6,err_msg=f"Unexpected solution for kernel weights")
        np.testing.assert_allclose(result.GetSolution(self.cstr.relax), np.zeros((1,)), atol=1e-6, err_msg=f"Unexpected relaxation variable solution")

    def test_get_slack(self):
        """Check that we can get the slack variables"""
        r = self.cstr.relax
        self.assertEqual(r.size, 1, msg='Unexpected shape for relaxation variables')

    def test_cost_weight(self):
        """Check that we can change the cost weight"""
        cost = self.cstr._ncp_cstr._slack_cost[0]
        test = np.ones((1,))
        np.testing.assert_allclose(cost.evaluator().Eval(test), np.ones((1,)), atol=1e-6, err_msg='Unexpected cost value before changing cost weight')
        self.cstr.cost_weight = 10
        np.testing.assert_allclose(cost.evaluator().Eval(test), 10*np.ones((1,)), atol=1e-6, err_msg='Unexpected cost value after changing cost weight')

class ContactTrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.traj = ce.ContactTrajectory()

    def test_set_generic(self):
        """Test the generic set_at method in contacttrajectory"""
        # Test changing an element that already exists
        test_list = [1, 2, 3, None, None, 6, 7]
        self.traj.set_at(test_list, 3, 4)
        self.assertEqual(test_list[3], 4, 'set_at failed to set the appropriate element in the test list')
        self.assertListEqual(test_list, [1, 2, 3, 4, None, 6, 7], 'set_at mutated other elements of the list')
        # Test changing an element that does not already exist
        self.traj.set_at(test_list, 9, 10)
        self.assertEqual(len(test_list), 10, msg="set_at fails to lengthen the list appropriately")
        self.assertEqual(test_list[9], 10, msg='set_at failed to add the new element to the list in the appropriate place')
        self.assertListEqual(test_list, [1, 2, 3, 4, None, 6, 7, None, None, 10], msg='set_at mutates the list when extending for additional values')

    def test_get_generic(self):
        """Test the generic get_at method in contacttrajectory"""
        test_list = [1, 2, 3, None, None, 6, 7]
        val = self.traj.get_at(test_list, 1)
        self.assertListEqual(val, [2], msg="get_at fails to return the correct single element list when only the start index is requested")
        val = self.traj.get_at(test_list, 1, 4)
        self.assertListEqual(val, [2, 3, None], msg="get_at fails to return the correct list when multiple indices are required")

    def test_add_time(self):
        """
        Check that the add_time works, that:
            the _time property is lengthed
            the num_timesteps changes after adding timepoints
        """
        test_times = [0.1, 0.3, 0.5]
        self.assertEqual(len(self.traj._time), 0, msg="time begins with a nonzero length")
        self.assertEqual(self.traj.num_timesteps, 0, msg='num_timesteps fails to return 0 when no timesteps have been added')
        for k, time in enumerate(test_times):
            self.traj.add_time(time)
            self.assertEqual(len(self.traj._time), k+1, msg="add_time did not lengthen the time array appropriately")
        self.assertEqual(self.traj.num_timesteps, 3, msg='num_timesteps does not return accurate results after updating the time vector')

        self.assertListEqual(self.traj._time, test_times, msg="calling add_times does not result in the correct time vector")

    def test_add_contact(self):
        """
        Check that the 'add_contact_sample' method works:
            1. the time vector is updated
            2. the number of contact points is updated
            3. getting the time and contact points afterwards returns the correct values
        """
        times = [0.2, 0.5]
        contacts = [np.array([[0.1, 0.2, 0.3], [0.1, 0.4, 0.3]]).T, np.array([[0.2, 0.3, 0.5], [0.4, 0.6, 0.8]]).T]
        for time, contact in zip(times, contacts):
            self.traj.add_contact_sample(time, contact)
        # Check that the points are correct
        self.assertEqual(self.traj.num_timesteps, 2, msg='add_contact_samples added the wrong number of timesteps')
        self.assertEqual(len(self.traj._contactpoints), 2, msg='add_contact_samples added the wrong number of contact points')
        # Get the contact points
        cpt = self.traj.get_contacts(0)
        np.testing.assert_array_equal(cpt[0], contacts[0], err_msg='get_contacts returns the wrong contact point at index 0')
        np.testing.assert_array_equal(self.traj.get_contacts(1)[0], contacts[1], err_msg='get_contacts returns the wrong contact point at index 1')

    def test_get_time_index(self):
        """
        Evaluate the getTimeIndex method of ContactTrajectory. Check that:
            1. getTimeIndex returns 0 when the sample point is less than the minimum time
            2. getTimeIndex returns the length of the time array when the sample point is more than the maximum time
            3. getTimeIndex returns the index of the nearest point when the sample is within the time bounds
            4. getTimeIndex returns the index of the nearest point when the exact time is given
        
        """
        # Add times to the time vector
        test_times = [0.1, 0.3, 0.5]
        for time in test_times:
            self.traj.add_time(time)
        # Check for a time before the first time
        self.assertEqual(self.traj.getTimeIndex(-0.1), 0, msg='getTimeIndex does not return 0 when the query time is less than the minimum time')
        # Check for times within the time vector
        self.assertEqual(self.traj.getTimeIndex(0.2), 0, msg='getTimeIndex fails to return 0 when the query time is greater than only the minimum time')
        self.assertEqual(self.traj.getTimeIndex(0.4), 1, msg='getTimeIndex fails to return the correct index when the query is within the time bounds')
        self.assertEqual(self.traj.getTimeIndex(0.3), 1, msg='getTimeIndex fails to return the correct index when the exact time is given')
        self.assertEqual(self.traj.getTimeIndex(0.5), 2, msg='getTimeIndex fails to return the last index with the exact time is given')
        # Check for a time outside the last time
        self.assertEqual(self.traj.getTimeIndex(0.6), 3, msg='getTimeIndex fails to return the length of the array when the query is greater than the maximum time')

class ContactEstimationTrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Setup the model for the class"""
        terrain = SemiparametricContactModel.FlatSurfaceWithRBFKernel(friction=0.5)
        cls.plant = Block(terrain = terrain)
        cls.plant.Finalize()
        cls.x0 = np.array([0., 0.5, 0., 0.])

    def setUp(self):
        """Setup the test examples and estimation trajectory container fresh for each test"""
        # Re-create the estimation trajectory fresh for each test
        self.traj = ce.ContactEstimationTrajectory(self.plant, self.x0)
        # Store the basic test example values
        self.x_test = np.array([0.1, 0.5, 1.0, 0.0]).T
        self.u_test = np.array([14.905])
        self.t_test = np.array([0.1])

    def test_append_sample(self):
        """
        Check that the "append_sample" method works:
            1. Check that the time vector is updated
            2. Check that the contact point vector is updated
            3. Check that the last_state is updated
            4. Check that the contact point added is correct
        """
        self.traj.append_sample(self.t_test, self.x_test, self.u_test)
        self.assertEqual(self.traj.num_timesteps, 2, msg='append_sample did not update time axis')
        # Just check that each list of constraints has been appropriately updated
        self.assertEqual(len(self.traj._contactpoints), 2, msg='append_sample did not update contactpoints list')
        # Check that the last_state has been updated
        np.testing.assert_equal(self.traj._last_state, self.x_test, err_msg="append_sample did not update last_state")
        # Check that the correct contact point was added
        cpt_expected = np.array([[0.1, 0.0, 0.0]]).T
        idx = self.traj.getTimeIndex(self.t_test)
        cpt = self.traj.get_contacts(idx)[0]
        np.testing.assert_allclose(cpt, cpt_expected, atol=1e-8, err_msg=f"Failed to return the correct contact point")

    def test_get_dynamics(self):
        """
        Test the dynamics constraints added by append_sample
        Append the test sample to the trajectory and test that:
            1. the appropriate number of dynamics constraints have been added
            2. getting the time index returns the appropriate set of dynamics constraints
            3. the dynamics parameters are the expected values
        """
        # Test values
        context = self.plant.multibody.CreateDefaultContext()
        self.plant.multibody.SetPositionsAndVelocities(context, self.x_test)
        J = self.plant.GetContactJacobians(context)
        A_expected = self.t_test * np.concatenate(J, axis=0).transpose()
        b_expected = np.array([-.4905, .9810])        
        # Append the sample
        self.traj.append_sample(self.t_test, self.x_test, self.u_test)
        self.assertEqual(len(self.traj._dynamics_cstr), 2, msg="append_sample did not update dynamics_cstr correctly")
        # Check the validity of the dynamics
        idx = self.traj.getTimeIndex(self.t_test)
        A, b = self.traj.getDynamicsConstraint(idx)
        np.testing.assert_allclose(A, A_expected, atol=1e-6, err_msg=f"incorrect contact jacobian in dynamics constraint")
        np.testing.assert_allclose(b, b_expected, atol=1e-6, err_msg=f"incorrect total contact force in dynamics constraint")

    def test_get_distance(self):
        """
        Test the distance constraint added by append_sample
        Append the test sample to the trajectory and test that:
            1. the appropriate number of distance constraints have been added
            2. getting the time index returns the appropriate set of distance constraints
            3. the distance parameters are the expected values
        """
        # Test values
        dist_expected = np.zeros((1,))       
        # Append the sample
        self.traj.append_sample(self.t_test, self.x_test, self.u_test)
        self.assertEqual(len(self.traj._distance_cstr), 2, msg="append_sample did not update distance constraints correctly")
        # Check the validity of the dynamics
        idx = self.traj.getTimeIndex(self.t_test)
        dist = self.traj.getDistanceConstraint(idx)
        np.testing.assert_allclose(dist, dist_expected, atol=1e-6, err_msg=f"incorrect distance constraint value")

    def test_get_dissipation(self):
        """
        Test the dissipation constraints added by append_sample
        Append the test sample to the trajectory and test that:
            1. the appropriate number of dissipation constraints have been added
            2. getting the time index returns the appropriate set of dissipation constraints
            3. the dissipation parameters are the expected values
        """
        # Test values
        A_expected = np.ones((4,1))
        context = self.plant.multibody.CreateDefaultContext()
        self.plant.multibody.SetPositionsAndVelocities(context, self.x_test)
        _, Jt = self.plant.GetContactJacobians(context)
        b_expected = Jt.dot(self.x_test[2:])    
        # Append the sample
        self.traj.append_sample(self.t_test, self.x_test, self.u_test)
        self.assertEqual(len(self.traj._dissipation_cstr), 2, msg="append_sample did not update dissipation_cstr correctly")
        # Check the validity of the dynamics
        idx = self.traj.getTimeIndex(self.t_test)
        A, b = self.traj.getDissipationConstraint(idx)
        np.testing.assert_allclose(A, A_expected, atol=1e-6, err_msg=f"incorrect duplication matrix in dissipation constraint")
        np.testing.assert_allclose(b, b_expected, atol=1e-6, err_msg=f"incorrect total tangential velocity in dissipation constraint")

    def test_get_friction(self):
        """
        Test the friction cone constraints added by append_sample
        Append the test sample to the trajectory and test that:
            1. the appropriate number of friction cone constraints have been added
            2. getting the time index returns the appropriate set of friction cone constraints
            3. the friction parameters are the expected values
        """
        # Test values
        A_expected = np.ones((1,4))
        b_expected = np.array([0.5])       
        # Append the sample
        self.traj.append_sample(self.t_test, self.x_test, self.u_test)
        self.assertEqual(len(self.traj._friction_cstr), 2, msg="append_sample did not update friction_cstr correctly")
        # Check the validity of the friction constraint
        idx = self.traj.getTimeIndex(self.t_test)
        A, b = self.traj.getFrictionConstraint(idx)
        np.testing.assert_allclose(A, A_expected, atol=1e-6, err_msg=f"incorrect duplication matrix in friction constraint")
        self.assertTrue(isinstance(b, list), 'returned friction coefficients should be a list')
        np.testing.assert_allclose(b[0], b_expected, atol=1e-6, err_msg=f"incorrect friction coefficient in frictio constraint")

    def test_get_force_guess(self):
        """
        Check that "get_force_guess" returns reasonable guesses for the reaction forces
            1. That the reaction force guess is nonnegative and the appropriate size when no other guess is given
            2. That the reaction force guess matches a provided guess when one is given
        """
        self.traj.append_sample(self.t_test, self.x_test, self.u_test)
        # Get force guess before any is given
        idx = self.traj.getTimeIndex(self.t_test)
        f_guess = self.traj.getForceGuess(idx)
        self.assertEqual(f_guess.shape, (5,), msg='default unexpected number of reaction forces in the guess')
        self.assertTrue(np.all(f_guess >= 0), msg='default force guess returns negative values for reaction forces')
        # Set the force guess and check that it is accurate
        f_test = np.array([1., 2., 3., 4., 5.])
        self.traj.set_force(idx, f_test)
        f_guess = self.traj.getForceGuess(idx)
        np.testing.assert_array_equal(f_guess, f_test, err_msg="force guess returned by getForceGuess does not match the value passed to set_force")

    def test_get_dissipation_guess(self):
        """
        Check that the 'get_dissipation_guess' returns reasonable guesses for the maximum dissipation velocity slack
            1. the dissipation slack is nonnegative and has the appropriate size
            2. that the dissipation slack matches a provided guess after one is given
        """
        self.traj.append_sample(self.t_test, self.x_test, self.u_test)
        # Get the dissipation guess before any is given
        idx = self.traj.getTimeIndex(self.t_test)
        d_guess = self.traj.getDissipationGuess(idx)
        self.assertEqual(d_guess.shape, (1,), msg='getDissipationGuess returns too many values')
        self.assertTrue(np.all(d_guess >= 0), msg='default dissipation guess is negative')
        # Set the dissipation
        test_val = np.array([2.])
        self.traj.set_dissipation(idx, test_val)
        d_guess = self.traj.getDissipationGuess(idx)
        np.testing.assert_array_equal(d_guess, test_val, err_msg="dissipation returnd by getDissipationGuess does not match the value passed to set_dissipation")

    def test_get_feasibility_guess(self):
        """
        Check that the 'get_feasibility_guess' returns reasonable guesses for the complementarity feasibility
            1. the feasibility is nonnegative when no other guess is provided
            2. the feasibility matches the provided guess when one is given
        """
        # Get the feasibility guess before any is given
        idx = self.traj.getTimeIndex(self.t_test)
        f_guess = self.traj.getFeasibilityGuess(idx)
        self.assertEqual(f_guess.shape, (1,), msg="default feasibility guess is nonscalar")
        self.assertTrue(f_guess.item() >= 0, msg="default feasibility guess is negative")
        # Set a feasibility guess and check that it's been set appropriately
        test_val = np.array([2.])
        self.traj.set_feasibility(idx, test_val)
        f_guess = self.traj.getFeasibilityGuess(idx)
        np.testing.assert_array_equal(f_guess, test_val, err_msg="feasibility returned by getFeasibilityGuess does not match the value passed to set_feasibility")

    def test_get_contact_kernels(self):
        """
        Check the getContactKernels method
            1. Evaluate the output when there are no points added
            2. Check that the two kernels have the correct shape (the number of contact points added) after add_sample is called
        """
        # Test before any samples are added
        Ks, Kf = self.traj.getContactKernels(0)
        np.testing.assert_allclose(Ks, np.eye(1), atol=1e-12, err_msg='surface kernel is the wrong value for one contact point')
        np.testing.assert_allclose(Kf, np.eye(1), atol=1e-12, err_msg='friction kernel is the wrong value for one contact point')
        # Test after any samples are added
        self.traj.append_sample(self.t_test, self.x_test, self.u_test)
        idx = self.traj.getTimeIndex(self.t_test)
        Ks, Kf = self.traj.getContactKernels(0, idx+1)
        self.assertEqual(Ks.shape, (2, 2), msg='surface kernel is the wrong shape after adding a second point')
        self.assertEqual(Kf.shape, (2, 2), msg="friction kernel is the wrong shape after adding a second point")

class ContactModelEstimatorTest(unittest.TestCase):
    def setUp(self):
        """
        Common setup for estimation tests
        """
        # Setup the semiparametric plant
        terrain = SemiparametricContactModel.FlatSurfaceWithRBFKernel(friction=0.5)
        plant = Block(terrain = terrain)
        plant.Finalize()
        x0 = np.array([0., 0.5, 0., 0.]).T
        # Setup the estimation trajectory
        self.traj = ce.ContactEstimationTrajectory(plant, x0)
        # Setup the estimator
        self.estimator = ce.ContactModelEstimator(self.traj, horizon=3)
        # Set some example values - example trajectory
        self.x_data = np.array([[0.1, 0.3, 0.6],
                                [0.5, 0.5, 0.5],
                                [1.0, 2.0, 3.0],
                                [0.0, 0.0, 0.0]])
        self.u_data = np.array([[14.95, 14.95, 14.95]])
        self.t_data = np.array([0.1, 0.2, 0.3])
        self.fN_data = np.array([[9.81, 9.81, 9.81, 9.81]])
        self.fT_data = np.zeros((4, 4))
        self.fT_data[2, 1:] = 4.95
        self.vs_data = np.array([[0., 1., 2., 3.]])
        # Append just the first point (bring total contacts to 2)
        self.traj.append_sample(self.t_data[0], self.x_data[:, 0], self.u_data[:, 0])

    def test_number_constraints(self):
        """
        Test that the program creates the correct number of constraints.
            1. When the total number of sample points is less than the desired horizon
            2. When the total number of sample points is more than the desired horizon
        """
        total_cstrs = lambda k: 10 * k + 1 
        self.estimator.create_estimator()
        self.assertEqual(len(self.estimator._prog.GetAllConstraints()), total_cstrs(2), 'unexpected number of constraints when there are fewer sample points than the desired horizon')
        # Add extra sample points
        self.traj.append_sample(self.t_data[1], self.x_data[:, 1], self.u_data[:, 1])
        self.traj.append_sample(self.t_data[2], self.x_data[:, 2], self.u_data[:, 2])
        # Remake and test the estimator
        self.estimator.create_estimator()
        self.assertEqual(len(self.estimator._prog.GetAllConstraints()), total_cstrs(3), "unexpected number of constraints when there are more sample points than the desired horizon")

    def test_number_costs(self): 
        """
        Test that the program creates the correct number of cost terms:
            1. When the total number of sample points is less than the desired horizon
            2. When the total number of sample points is more than the desired horizon
        """
        total_costs = lambda k: 4
        self.estimator.create_estimator()
        self.assertEqual(len(self.estimator._prog.GetAllCosts()), total_costs(2), "unexpected number of costs when there are fewer sample points than the desired horizon")
        # Add sample points
        self.traj.append_sample(self.t_data[1], self.x_data[:, 1], self.u_data[:, 1])
        self.traj.append_sample(self.t_data[2], self.x_data[:, 2], self.u_data[:, 2])
        # Check the number of costs now
        self.estimator.create_estimator()
        self.assertEqual(len(self.estimator._prog.GetAllCosts()), total_costs(3), "Unexpected number of costs when there are more sample points than the desired horizon")

    def test_number_variables(self):
        """
        Test that the program creates the correct number of decision variables:
            1. When the total number of sample points is less than the desired horizon
            2. When the total number of sample points is more than the desired horizon
        """
        c = 5 * self.traj.num_contacts + 2 * self.traj.num_friction  + 1 
        total_vars = lambda k : c * k
        self.estimator.create_estimator()
        self.assertEqual(self.estimator._prog.decision_variables().size, total_vars(2), 'unexpected number of decision variables when there are fewer sample points than the desired horizon')
        # Add extra sample points
        self.traj.append_sample(self.t_data[1], self.x_data[:, 1], self.u_data[:, 1])
        self.traj.append_sample(self.t_data[2], self.x_data[:, 2], self.u_data[:, 2])
        # Remake and test the estimator
        self.estimator.create_estimator()
        self.assertEqual(self.estimator._prog.decision_variables().size, total_vars(3), "unexpected number of decision variables when there are more sample points than the desired horizon")

    def test_solve(self):
        """
        Test that the program can solve successfully and returns a reasonable solution (given the solution)
        """
        # Setup the program
        self.estimator.create_estimator()
        # Set the initial guess for the decision variables to their expected values
        dweight_guess = np.zeros(self.estimator._distance_weights.shape)
        fweight_guess = np.zeros(self.estimator._friction_weights.shape)
        f_guess = np.vstack([self.fN_data[:, :2], self.fT_data[:, :2]])
        sV_guess = self.vs_data[:, :2]
        r_guess = np.zeros((1,2))   
        self.estimator._prog.SetInitialGuess(self.estimator._distance_weights, dweight_guess)
        self.estimator._prog.SetInitialGuess(self.estimator._friction_weights, fweight_guess)
        self.estimator._prog.SetInitialGuess(self.estimator.forces, f_guess)
        self.estimator._prog.SetInitialGuess(self.estimator.velocities, sV_guess)
        self.estimator._prog.SetInitialGuess(self.estimator.feasibilities, r_guess)
        # Solve the program
        self.estimator.setSolverOptions({'Major feasibility tolerance': 1e-8,
                                        'Major optimality tolerance': 1e-8})
        result = self.estimator.solve()
        self.assertTrue(result.is_success(), "Failed to solve the estimation problem successfully")
        # Check the answers against the initial guesses
        with self.subTest(msg='Distance Solution'):
            np.testing.assert_allclose(result.GetSolution(self.estimator._distance_weights), dweight_guess, atol=1e-6, err_msg=f'Distance weight solution inaccurate')
        with self.subTest(msg='Friction Solution'):
            np.testing.assert_allclose(result.GetSolution(self.estimator._friction_weights), 
            fweight_guess, atol=1e-6, err_msg=f'Friction weight solution inaccurate')
        with self.subTest(msg='Force Solution'):
            np.testing.assert_allclose(result.GetSolution(self.estimator.forces),
            f_guess, atol=1e-6, err_msg=f"Force solution inaccurate")
        with self.subTest(msg='Velocity solution'):
            np.testing.assert_allclose(result.GetSolution(self.estimator.velocities), sV_guess, atol=1e-6, err_msg="Velocity slack solution inaccurate")
        with self.subTest(msg='Feasibility Solution'):        
            np.testing.assert_allclose(result.GetSolution(self.estimator.feasibilities), r_guess, atol=1e-6, err_msg=f"Feasibility solution inaccurate")

    def test_estimate(self):
        """
        Test that the wrapper method, estimate_contact, successfully solves the problem and stores the solution in the "guesses" of the contactestimationtrajectory.
        """
        # Run estimation
        model = self.estimator.estimate_contact(self.t_data[1], self.x_data[:, 1], self.u_data[:, 1])
        # Check that the returned model is a Semiparametric Contact Model
        self.assertTrue(isinstance(model, SemiparametricContactModel), msg=f"Returned value is not a SemiparametricContactModel")
        # Check that contact trajectory has been appropriately updated
        idx = self.traj.getTimeIndex(self.t_data[1])
        self.assertEqual(len(self.traj._forces), idx+1, 'calling estimate_contact results in an unexpected number of stored reaction forces')
        self.assertEqual(len(self.traj._slacks), idx+1, 'calling estimate_contact results in an unexpected number of stored velocity slacks')
        self.assertEqual(len(self.traj._feasibility), idx+1, 'calling estimate_conact results in an unexpected number of feasibility variables')


if __name__ == '__main__':
    unittest.main()