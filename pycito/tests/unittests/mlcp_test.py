"""
Unittests for the mixed linear complementarity constraints in controller.mlcp.py

For mixed linear constraints, setup and solves the following problem:
    min (x - d)^2
        Ax + Bz + c >= 0
        z >= 0
        z * (Ax + Bz + c) = 0
where A, B, c, d are constants.

For the A, B, c, d specified in the program, the optimal solution is:
    x = [2, -1]
    z = [0,  1]

for pseudo-linear constraints, setup and sole the following problem:
    min (x - d)^2 + (z - b)^2
        Ax + c >=0
        z >= 0
        z * (Ax + c) = 0
    where A, b, c, d are constants

For A, b, c, d specified, the optimal solution is:
    x = [0, 1]
    z = [0, 2]

Luke Drnach
January 27, 2022
"""

import unittest
import numpy as np
from pycito.controller import mlcp
from pydrake.all import MathematicalProgram, Solve

class _LCPTestBase():
    """
    Contains all common tests for LCP problems
    """
    def setUp(self):
        """
        subclasses should add a concrete example. This method just does a bare-bones setup
        """
        # Create the program and associated variables
        self.prog = MathematicalProgram()
        self.x = self.prog.NewContinuousVariables(rows=2,  name='x')
        self.z = self.prog.NewContinuousVariables(rows=2,  name='z')
        # Add the complementarity constraints
        self.cstr = None
        self.setup_complementarity_constraints()
        self.cstr.addToProgram(self.prog, self.x, self.z)
        # list the expected outputs
        self.expected_num_variables = 6
        self.expected_num_constraints = 3
        self.expected_num_costs = 1
        self.expected_num_slacks = 2
        # Store optimal solution
        self.x_expected = np.array([2, -1])
        self.z_expected = np.array([0, 1])

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
        self.assertEqual(self.cstr.slack.size, self.expected_num_slacks, msg="Unexpected number of slack variables")

    def test_solve_problem(self):
        """Test that we can solve the problem, at least given the optimal solution"""
        result = self.solve_problem()
        self.assertTrue(result.is_success(), msg=f"Failed to successfully solve with solver {result.get_solver_id().name()}")
        self.check_result(result)

class _MLCPExampleMixin(_LCPTestBase):
    """
    Test the implementation of mixed linear complementarity problems
    """
    def setUp(self):
        # Complementarity constraint constannts
        self.A = np.array([[1, 2], [-1, 0]])
        self.B = np.array([[2,1],[0,2]])
        self.c = np.array([-1, 0])
        # Setup the main program
        super(_MLCPExampleMixin, self).setUp()
        # Add a cost to regularize the problem
        Q = np.eye(2)
        b = np.array([2, -1])
        self.prog.AddQuadraticErrorCost(Q, b, vars=self.x)
        # Store optimal solution
        self.x_expected = np.array([2, -1])
        self.z_expected = np.array([0, 1])
        # Set initial guess
        self.prog.SetInitialGuess(self.x, self.x_expected)
        self.prog.SetInitialGuess(self.z, self.z_expected)

    def test_initialize_slacks(self):
        """Test that we can correctly initialize the slack variables"""
        self.cstr.initializeSlackVariables()
        svals = self.prog.GetInitialGuess(self.cstr._slack)
        s_expected = self.A.dot(self.x_expected) + self.B.dot(self.z_expected) + self.c
        np.testing.assert_allclose(svals, s_expected, atol=1e-6, err_msg = "slack variables initialized incorrectly")

    def test_random(self):
        """Test random initialization of the constraint"""
        cstr_type = type(self.cstr)
        xdim = 2
        zdim = 3
        rand_cstr = cstr_type.random(xdim, zdim)
        self.assertTrue(type(rand_cstr) is cstr_type, f"Random does not produce the correct type")
        self.assertEqual(rand_cstr.A.shape, (zdim, xdim), f"Random A matrix has incorrect shape")
        self.assertEqual(rand_cstr.B.shape, (zdim, zdim), f"Random B matrix has incorrect shape")
        self.assertEqual(rand_cstr.c.shape, (zdim,), f"Random c vector has incorrect shape")

    def test_update_coefficients(self):
        """Test that we can """
        lincstr = self.cstr._lincstr.evaluator()
        A = 2*self.A
        B = 3*self.B
        c = -1*self.c
        self.cstr.updateCoefficients(A, B, c)
        A_new = np.concatenate([np.eye(self.cstr.dim), -A, -B], axis=1)
        np.testing.assert_allclose(lincstr.A(), A_new, atol=1e-8, err_msg=f"calling UpdateCoefficients did not update linear constraint coefficients")
        np.testing.assert_allclose(lincstr.lower_bound(), c, atol=1e-8, err_msg=f"calling UpdateCoefficients did not update the lower bound on the linear constraint")
        np.testing.assert_allclose(lincstr.upper_bound(), c, atol=1e-8, err_msg=f"calling UpdateCoefficients did not update the upper bound on the linear constraint")
        np.testing.assert_allclose(self.cstr.A, A, atol=1e-8, err_msg=f"updateCoefficients did not update the coefficient A in the LCP object")
        np.testing.assert_allclose(self.cstr.B, B, atol=1e-8, err_msg=f"updateCoefficients did not update the coefficient A in the LCP object")
        np.testing.assert_allclose(self.cstr.c, c, atol=1e-8, err_msg=f"updateCoefficients did not update the coefficient A in the LCP object")

class MixedLinearComplementarityTest(_MLCPExampleMixin, unittest.TestCase):
    def setUp(self):
        super(MixedLinearComplementarityTest, self).setUp()

    def setup_complementarity_constraints(self):
        self.cstr = mlcp.MixedLinearComplementarityConstraint(self.A, self.B, self.c)

class CostRelaxedMLCPTest(_MLCPExampleMixin, unittest.TestCase):
    def setUp(self):
        super(CostRelaxedMLCPTest, self).setUp()
        self.expected_num_constraints -= 1
        self.expected_num_costs += 1

    def setup_complementarity_constraints(self):
        self.cstr = mlcp.CostRelaxedMixedLinearComplementarity(self.A, self.B, self.c)

class VariableRelaxedMLCPTest(_MLCPExampleMixin, unittest.TestCase):
    def setUp(self):
        super(VariableRelaxedMLCPTest, self).setUp()
        self.expected_num_constraints += 1
        self.expected_num_costs += 1
        self.expected_num_variables += 1
        self.expected_num_slacks += 1  

    def setup_complementarity_constraints(self):
        self.cstr = mlcp.VariableRelaxedMixedLinearComplementarityConstraint(self.A, self.B, self.c)

    def test_update_cost(self):
        """
            Test that we can change the cost weight and it's reflected in the associated cost
        """
        self.cstr.cost_weight = 10
        cost = self.cstr._cost.evaluator().Eval(np.ones((1,)))
        np.testing.assert_allclose(cost, 10*np.ones((1,)), atol=1e-6, err_msg="Relaxation cost incorrectly evaluated after updating")


class _PseudoLinearExampleMixin(_LCPTestBase):
    """
        Tests the implementation of various pseudo-linear complementarity constraints
    """
    def setUp(self):
        # Create the PLCP example case
        self.A = np.array([[1, 2], [-1, 0]])
        self.c = np.array([-1, 0])
        super(_PseudoLinearExampleMixin, self).setUp()
        # Store the optimal solutions
        self.x_expected = np.array([0, 1])
        self.z_expected = np.array([0, 2])
        # Add a cost to regularize the problem
        self.prog.AddQuadraticErrorCost(np.eye(2), self.x_expected, self.x)
        self.prog.AddQuadraticErrorCost(np.eye(2), self.z_expected, self.z)
        # Initialize 
        self.prog.SetInitialGuess(self.x, self.x_expected)
        self.prog.SetInitialGuess(self.z, self.z_expected)
        # Update the expected number of costs
        self.expected_num_costs = 2

    def test_initialize_slack(self):
        """Test that we can initialize the slack variables"""
        self.cstr.initializeSlackVariables()
        svals = self.prog.GetInitialGuess(self.cstr._slack)
        s_expected = self.A.dot(self.x_expected) + self.c
        np.testing.assert_allclose(svals, s_expected, atol=1e-6, err_msg="Slack variables incorrectly initialized")

    def test_update_coefficients(self):
        """Test that we can """
        lincstr = self.cstr._lincstr.evaluator()
        A = 2*self.A
        c = -1*self.c
        self.cstr.updateCoefficients(A, c)
        A_new = np.concatenate([np.eye(self.cstr.dim), -A], axis=1)
        np.testing.assert_allclose(lincstr.A(), A_new, atol=1e-8, err_msg=f"calling UpdateCoefficients did not update linear constraint coefficients")
        np.testing.assert_allclose(lincstr.lower_bound(), c, atol=1e-8, err_msg=f"calling UpdateCoefficients did not update the lower bound on the linear constraint")
        np.testing.assert_allclose(lincstr.upper_bound(), c, atol=1e-8, err_msg=f"calling UpdateCoefficients did not update the upper bound on the linear constriant")
        np.testing.assert_allclose(self.cstr.A, A, atol=1e-8, err_msg=f"updateCoefficients did not update the coefficient A in the LCP object")
        np.testing.assert_allclose(self.cstr.c, c, atol=1e-8, err_msg=f"updateCoefficients did not update the coefficient A in the LCP object")

    def test_random(self):
        """Test random initialization of the constraint"""
        cstr_type = type(self.cstr)
        xdim = 2
        zdim = 3
        rand_cstr = cstr_type.random(xdim, zdim)
        self.assertTrue(type(rand_cstr) is cstr_type, f"Random does not produce the correct type")
        self.assertEqual(rand_cstr.A.shape, (zdim, xdim), f"Random A matrix has incorrect shape")
        self.assertEqual(rand_cstr.c.shape, (zdim,), f"Random c vector has incorrect shape")

class PseudoLinearComplementarityTest(_PseudoLinearExampleMixin, unittest.TestCase):
    def setUp(self):
        super(PseudoLinearComplementarityTest, self).setUp()

    def setup_complementarity_constraints(self):
        self.cstr = mlcp.PseudoLinearComplementarityConstraint(self.A, self.c)


class CostRelaxedPseudoLinearComplementarityTest(_PseudoLinearExampleMixin,unittest.TestCase):
    def setUp(self):
        super(CostRelaxedPseudoLinearComplementarityTest, self).setUp()
        self.expected_num_constraints -= 1
        self.expected_num_costs += 1

    def setup_complementarity_constraints(self):
        self.cstr = mlcp.CostRelaxedPseudoLinearComplementarityConstraint(self.A, self.c)

class VariableRelaxedPseudoLinearComplementarityTest(_PseudoLinearExampleMixin, unittest.TestCase):
    def setUp(self):
        super(VariableRelaxedPseudoLinearComplementarityTest, self).setUp()
        self.expected_num_constraints += 1
        self.expected_num_costs += 1
        self.expected_num_variables += 1
        self.expected_num_slacks += 1

    def setup_complementarity_constraints(self):
        self.cstr = mlcp.VariableRelaxedPseudoLinearComplementarityConstraint(self.A, self.c)

    def test_update_cost(self):
        """
            Test that we can change the cost weight and it's reflected in the associated cost
        """
        self.cstr.cost_weight = 10
        cost = self.cstr._cost.evaluator().Eval(np.ones((1,)))
        np.testing.assert_allclose(cost, 10*np.ones((1,)), atol=1e-6, err_msg="Relaxation cost incorrectly evaluated after updating")

if __name__ == '__main__':
    unittest.main()