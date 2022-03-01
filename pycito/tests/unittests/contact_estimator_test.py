import numpy as np
import unittest

from pydrake.all import MathematicalProgram, Solve

from pycito.controller import contactestimator as ce
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
    pass

class ContactModelEstimatorTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()