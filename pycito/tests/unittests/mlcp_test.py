"""
Unittests for the mixed linear complementarity constraints in controller.mlcp.py

Setup and solves the following problem:
    min (x - d)^2
        Ax + Bz + c >= 0
        z >= 0
        z * (Ax + Bz + c) = 0
where A, B, c, d are constants.

For the A, B, c, d specified in the program, the optimal solution is:
    x = [2, -1]
    z = [0,  1]

Luke Drnach
January 27, 2022
"""

import unittest
import numpy as np
from pycito.controller import mlcp
from pydrake.all import MathematicalProgram, Solve

class _MLCPTestBaseMixin():
    """
    Test the implementation of mixed linear complementarity problems
    """
    def setUp(self):
        # Complementarity constraint constannts
        self.A = np.array([[1, 2], [-1, 0]])
        self.B = np.array([[2,1],[0,2]])
        self.c = np.array([-1, 0])
        # Create the program and associated variables
        self.prog = MathematicalProgram()
        self.x = self.prog.NewContinuousVariables(rows=2,  name='x')
        self.z = self.prog.NewContinuousVariables(rows=2,  name='z')
        # Add the complementarity constraints
        self.cstr = None
        self.setup_complementarity_constraints()
        self.cstr.addToProgram(self.prog, self.x, self.z)
        # Add a cost to regularize the problem
        Q = np.eye(2)
        b = np.array([2, -1])
        self.prog.AddQuadraticErrorCost(Q, b, vars=self.x)
        # Expected test outputs
        self.expected_num_variables = 6
        self.expected_num_constraints = 3
        self.expected_num_costs = 1
        self.expected_num_slacks = 2
        # Store optimal solution
        self.x_expected = np.array([2, -1])
        self.z_expected = np.array([0, 1])
        # Set initial guess
        self.prog.SetInitialGuess(self.x, self.x_expected)
        self.prog.SetInitialGuess(self.z, self.z_expected)

    def setup_complementarity_constraints(self):
        raise NotImplementedError()

    def solve_problem(self):
        """Helper function to solve the complementarity problem"""
        result = Solve(self.prog)
        self.assertTrue(result.is_success(), msg=f"Failed to solve complementarity problem with {type(self.cstr).__name__}")
        return result

    def check_result(self, result):
        """Check that the result is the expected solution"""
        x_ = result.GetSolution(self.x)
        z_ = result.GetSolution(self.z)
        np.testing.assert_allclose(x_, self.x_expected, atol=1e-3, err_msg=f"Free variable solution not close enough using {type(self.cstr).__name__}")
        np.testing.assert_allclose(z_, self.z_expected, atol=1e-3, err_msg=f"Complementarity variable solution not close enough using {type(self.cstr).__name__}")

    def test_constraints_added(self):
        """Check that the program has the expected number of constraints"""
        self.assertEqual(len(self.prog.GetAllConstraints()), self.expected_num_constraints, msg='Unexpected number of constraints')

    def test_costs_added(self):
        """Check that the program has the expected number of costs"""
        self.assertEqual(len(self.prog.GetAllCosts()), self.expected_num_costs, msg="Unexpected number of cost terms")

    def test_total_variables(self):
        """Check that the program has the expected number of variables"""
        self.assertEqual(self.prog.num_vars(), self.expected_num_variables, msg="Unexpected number of decision variables")

    def test_get_slacks(self):
        """Check that the program sets the expected number of slack variables"""
        self.assertEqual(self.cstr.var_slack.size, self.expected_num_slacks, msg="Unexpected number of slack variables")

    def test_solve_problem(self):
        """Test that we can solve the problem, at least given the optimal solution"""
        result = self.solve_problem()
        self.check_result(result)

class MixedLinearComplementarityTest(_MLCPTestBaseMixin, unittest.TestCase):
    def setUp(self):
        super(MixedLinearComplementarityTest, self).setUp()

    def setup_complementarity_constraints(self):
        self.cstr = mlcp.MixedLinearComplementarityConstraint(self.A, self.B, self.c)

class CostRelaxedMLCPTest(_MLCPTestBaseMixin, unittest.TestCase):
    def setUp(self):
        super(CostRelaxedMLCPTest, self).setUp()
        self.expected_num_constraints -= 1
        self.expected_num_costs += 1

    def setup_complementarity_constraints(self):
        self.cstr = mlcp.CostRelaxedMixedLinearComplementarity(self.A, self.B, self.c)

if __name__ == '__main__':
    unittest.main()