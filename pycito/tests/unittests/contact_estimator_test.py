import numpy as np
import unittest

from pydrake.all import MathematicalProgram, Solve

from pycito.controller import contactestimator as ce

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
        self.cstr.addToProgram(self.prog, self.w, self.fN, self.fT, self.vS)
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


if __name__ == '__main__':
    unittest.main()