"""
Unittests for the complementarity constraints in trajopt.complementarity

Set-up and solve the problem:
min x_1 + x_2
s.t.
y - (x_1 - x_2)^2 = 1
x_1, x_2 >=0
x_1*x_2 = 0  

The optimal solution is (x_1, x_2, y) = (0, 0, 1)

Luke Drnach
June 14, 2021
"""

import unittest
import numpy as np
from pycito.trajopt import complementarity as cp
from pydrake.all import MathematicalProgram, Solve

def pass_through(x):
    return x

def example_constraint(z):
    x1, x2, y = np.split(z, 3)
    return y - (x1 - x2)**2

def test_constraint(x):
    return x - 1

class _ComplementarityBaseTestMixin():
    """
    Test the implementation of complementarity constraints with the example problem:
        min (z-2)**2
        x - 1 >= 0
        z >= 0
        z*(x-1) = 0

    The solution should be (x, z) = (1, 2)
    """
    def setUp(self):
        # Create the program and associated variables
        self.prog = MathematicalProgram()
        self.fcn = test_constraint
        self.x = self.prog.NewContinuousVariables(rows=2, cols=1, name='x')
        self.z = self.prog.NewContinuousVariables(rows=2, cols=1, name='z')
        # Setup the complementarity constraint
        self.setup_complementarity_constraints()
        self.cstr.addToProgram(self.prog, xvars = self.x, zvars = self.z)
        # Add a cost to the program to regularize it
        Q = np.eye(2)
        b = 2 * np.ones((2,))
        self.prog.AddQuadraticErrorCost(Q, b, vars=self.z)
        # Expected test output values
        self.expected_num_variables = 4
        self.expected_num_constraints = 1
        self.expected_num_costs = 1
        self.expected_num_slacks = 0

        # Store the optimal solution
        self.x_expected = np.ones((2,))
        self.z_expected = 2*np.ones((2,))
        
        # Set the initial guess for the program - initialize at the solution
        self.prog.SetInitialGuess(self.x, self.x_expected)
        self.prog.SetInitialGuess(self.z, self.z_expected)

    def setup_complementarity_constraints(self):
        raise NotImplementedError()

    def solve_problem(self):
        """ Helper function to solve the complementarity problem """
        # Check that the problem solves successfully
        result = Solve(self.prog)
        self.assertTrue(result.is_success(), msg=f"Failed to solve complementarity problem with {type(self.cstr).__name__}")
        return result

    def check_result(self, result):
        """ Check that the results are reasonably accurate"""
        x_ = result.GetSolution(self.x)
        z_ = result.GetSolution(self.z)
        np.testing.assert_allclose(x_, self.x_expected, atol= 1e-3, err_msg=f"Results are not close to the expected solution using {type(self.cstr).__name__}")
        np.testing.assert_allclose(z_, self.z_expected, atol=1e-3, err_msg =f"Results are not close to the expected solution using {type(self.cstr).__name__}")

    def get_constraint_upper_bound(self):
        """ Get the upper bound for the complementarity constraint"""
        for cstr in self.prog.GetAllConstraints():
            if cstr.evaluator().get_description() == self.fcn.__name__:
                return cstr.evaluator().upper_bound()

    def check_constant_slack(self):
        """ Test snippet for checking if setting a constant slack changes the upper bounds"""
        ub_1 = self.get_constraint_upper_bound()
        self.cstr.const_slack = 1.
        ub_2 = self.get_constraint_upper_bound()
        self.assertTrue(np.any(np.not_equal(ub_1, ub_2)), msg=f"Changing constant slack does not change the upper bound for {type(self.cstr).__name__}")

    def check_setting_cost(self, result):
        """ Test snippet for checking if setting a cost weight works"""
        self.cstr.cost_weight = 1.
        cost1 = self.eval_complementarity_cost(result)
        self.cstr.cost_weight = 10.
        cost2 = self.eval_complementarity_cost(result)
        self.assertAlmostEqual(cost1*10, cost2, delta=1e-7, msg=f"Setting cost_weight does not change cost for {type(self.cstr).__name__}")

    def eval_complementarity_cost(self, result):
        """ Evaluate the cost associated with the complementarity constraint"""
        costs = self.prog.GetAllCosts()
        for cost in costs:
            if cost.evaluator().get_description() in [f"{self.fcn.__name__}SlackCost", f"{self.fcn.__name__}Cost"]:
                dvars = cost.variables()
                dvals = result.GetSolution(dvars)
                break
        return cost.evaluator().Eval(dvals) 

    def test_constraints_added(self):
        """Check that the program has the correct number of constraints added"""
        self.assertEqual(len(self.prog.GetAllConstraints()), self.expected_num_constraints, msg = 'Unexpected number of constraints')

    def test_costs_added(self):
        """Check that the program has the expected number of costs added"""
        self.assertEqual(len(self.prog.GetAllCosts()), self.expected_num_costs, msg='Unexpected number of costs')

    def test_get_slacks(self):
        """Check that the correct number of slack variables has been added"""

        # Note: Constant slack methods return None as the var_slack, 
        if self.cstr.var_slack is not None:
            self.assertEqual(self.cstr.var_slack.size, self.expected_num_slacks, msg = 'Unexpected number of slack variables')
        else:
            self.assertEqual(0, self.expected_num_slacks, msg='Unexpected number of slack variables')

    def test_number_variables(self):
        """Check the total number of variables"""
        self.assertEqual(self.prog.num_vars(), self.expected_num_variables, msg="Unexpected number of decision variables")

    def test_solve_problem(self):
        """ Test that solving the problem achieves the desired solution """
        result = self.solve_problem()
        self.check_result(result)


class NonlinearComplementarityConstantSlackTest(_ComplementarityBaseTestMixin, unittest.TestCase):
    def setUp(self):
        super(NonlinearComplementarityConstantSlackTest, self).setUp()

    def setup_complementarity_constraints(self):
        self.cstr = cp.NonlinearConstantSlackComplementarity(self.fcn, xdim = 2, zdim = 2)

    def test_constant_slack(self):
        """Test that setting constant slack variables changes the program"""
        self.check_constant_slack()
        
class NonlinearComplementarityVariableSlackTest(_ComplementarityBaseTestMixin, unittest.TestCase):
    def setUp(self):
        super(NonlinearComplementarityVariableSlackTest, self).setUp()
        self.expected_num_slacks += 1
        self.expected_num_variables += 1
        self.expected_num_costs += 1

    def setup_complementarity_constraints(self):
        self.cstr = cp.NonlinearVariableSlackComplementarity(self.fcn, xdim = 2, zdim = 2)

    def test_cost_weight(self):
        """Test if setting the cost weight has any effect"""
        result = self.solve_problem()
        self.check_setting_cost(result)

class NonlinearComplementarityWithCostTest(_ComplementarityBaseTestMixin, unittest.TestCase):
    def setUp(self):
        super(NonlinearComplementarityWithCostTest, self).setUp()
        self.expected_num_costs += 1

    def setup_complementarity_constraints(self):
        self.cstr = cp.CostRelaxedNonlinearComplementarity(self.fcn, xdim=2, zdim=2)

    def test_cost_weight(self):
        """Test if setting the cost weight has any effect"""
        result = self.solve_problem()
        self.check_setting_cost(result)

class LinearComplementarityConstantSlackTest(_ComplementarityBaseTestMixin, unittest.TestCase):
    def setUp(self):
        super(LinearComplementarityConstantSlackTest, self).setUp()
        self.expected_num_slacks += 2
        self.expected_num_variables += 2

    def setup_complementarity_constraints(self):
        self.cstr = cp.LinearEqualityConstantSlackComplementarity(self.fcn, xdim=2, zdim=2)

    def test_constant_slack(self):
        """Test that setting constant slack variables changes the program"""
        self.check_constant_slack()

class LinearComplementarityVariableSlackTest(_ComplementarityBaseTestMixin, unittest.TestCase):
    def setUp(self):
        super(LinearComplementarityVariableSlackTest, self).setUp()
        self.expected_num_slacks += 3
        self.expected_num_variables += 3
        self.expected_num_costs += 1

    def setup_complementarity_constraints(self):
        self.cstr = cp.LinearEqualityVariableSlackComplementarity(self.fcn, xdim=2, zdim=2)

    def test_cost_weight(self):
        """Test if setting the cost weight has any effect"""
        result = self.solve_problem()
        self.check_setting_cost(result)

class LinearComplementarityWithCostTest(_ComplementarityBaseTestMixin, unittest.TestCase):
    def setUp(self):
        super(LinearComplementarityWithCostTest, self).setUp()
        self.expected_num_slacks += 2
        self.expected_num_variables += 2
        self.expected_num_costs += 1

    def setup_complementarity_constraints(self):
        self.cstr = cp.CostRelaxedLinearEqualityComplementarity(self.fcn, xdim = 2, zdim = 2)

    def test_cost_weight(self):
        """Test if setting the cost weight has an effect"""
        result = self.solve_problem()
        self.check_setting_cost(result)

class CollocatedComplementarityBaseTestMixin():
    def setUp(self):
        # Complementarity function
        self.fcn = test_constraint
        # Setup the problem
        self.prog = MathematicalProgram()
        self.x = self.prog.NewContinuousVariables(rows = 2, cols = 3, name = 'x')
        self.z = self.prog.NewContinuousVariables(rows = 2, cols = 3, name = 'z')
        self.add_complementarity_constraints()
        # Add a cost to the program to regularize it
        Q = np.eye(6)
        b = 2 * np.ones((6,))
        self.prog.AddQuadraticErrorCost(Q, b, vars=self.z.flatten())
        # Expected test output values
        self.expected_num_variables = 16
        self.expected_num_constraints = 9
        self.expected_num_costs = 1
        self.expected_num_slacks = 4
        # Set the initial guess for the program - initialize at the solution
        self.prog.SetInitialGuess(self.x, np.ones((2, 3)))
        self.prog.SetInitialGuess(self.z, 2*np.ones((2, 3)))

    def add_complementarity_constraints(self):
        """Add complementarity constraint - overwritten in subclasses"""
        self.cstr = cp.CollocatedComplementarity(self.fcn, self.x.shape[0], self.z.shape[0])

    def solve_problem(self):
        """Solve the complementarity problem"""
        result = Solve(self.prog)
        self.assertTrue(result.is_success(), msg=f"Failed to solve complementarity problem with {type(self.cstr).__name__}")
        return result

    def test_decision_variables(self):
        """Check that the test has the appropriate number of decision variables"""
        self.assertEqual(self.prog.num_vars(), self.expected_num_variables, msg="Unexpected number of decision variables")

    def test_constraints_added(self):
        """Check that the program has the correct number of constraints added"""
        self.assertEqual(len(self.prog.GetAllConstraints()), self.expected_num_constraints, msg = 'Unexpected number of constraints')

    def test_costs_added(self):
        """Check that the program has the expected number of costs added"""
        self.assertEqual(len(self.prog.GetAllCosts()), self.expected_num_costs, msg='Unexpected number of costs')

    def test_get_slacks(self):
        """Check that the correct number of slack variables has been added"""
        self.assertEqual(self.cstr.var_slack.size, self.expected_num_slacks, msg = 'Unexpected number of slack variables')

    def test_solution_result(self):
        """Check that the problem solves and we can get a solution"""
        result = self.solve_problem()
        xval = result.GetSolution(self.x)
        zval = result.GetSolution(self.z)
        np.testing.assert_allclose(xval, np.ones((2, 3)), err_msg = 'Solution for x is incorrect')
        np.testing.assert_allclose(zval, 2*np.ones((2, 3)), err_msg = 'Solution for z is incorrect')

    def test_collocation_variables(self):
        """Check that the collocation variables are correct"""
        result = self.solve_problem()
        slacks = self.cstr.var_slack
        slack_vals = result.GetSolution(slacks)
        expected = np.array([0., 0., 6., 6.])
        np.testing.assert_allclose(slack_vals[:4], expected, err_msg = "Solved values for collocation slack variables is incorrect")

class CollocatedComplementarityConstantSlackTest(CollocatedComplementarityBaseTestMixin, unittest.TestCase):
    def setUp(self):
        super(CollocatedComplementarityConstantSlackTest, self).setUp()
    
    def add_complementarity_constraints(self):
        """Add collocated complementarity constraint to the program"""
        self.cstr = cp.CollocatedConstantSlackComplementarity(self.fcn, self.x.shape[0], self.z.shape[0])
        self.cstr.addToProgram(self.prog, self.x, self.z)

    def get_orthogonality_bounds(self):
        for c in self.prog.GetAllConstraints():
            if c.evaluator().get_description() == self.cstr.name + '_orthogonality':
                return c.evaluator().lower_bound(), c.evaluator().upper_bound()

    def test_setting_slack(self):
        """Test setting a constant slack changes the upper bound on the orthogonality constraint"""
        self.cstr.const_slack = 0
        lb_pre, ub_pre = self.get_orthogonality_bounds()
        self.cstr.const_slack = 1
        lb_post, ub_post = self.get_orthogonality_bounds()
        np.testing.assert_allclose(lb_post, lb_pre, err_msg='Lower bounds changed when adding slack')
        np.testing.assert_allclose(ub_pre,  np.zeros((self.z.shape[0],)), err_msg = 'Upper bounds incorrect before adding slack')
        np.testing.assert_allclose(ub_post, np.ones((self.z.shape[0],)), err_msg = 'Upper bounds incorrect after adding slack')

class CollocatedComplementarityVariableSlackTest(CollocatedComplementarityBaseTestMixin, unittest.TestCase):
    def setUp(self):
        super(CollocatedComplementarityVariableSlackTest, self).setUp()
        # Update expected problem parameters
        self.expected_num_variables += 1
        self.expected_num_costs += 1
        self.expected_num_slacks += 1

    def add_complementarity_constraints(self):
        """Add collocated complementarity constraint to the program"""
        self.cstr = cp.CollocatedVariableSlackComplementarity(self.fcn, self.x.shape[0], self.z.shape[0])
        self.cstr.addToProgram(self.prog, self.x, self.z)

    def eval_complementarity_cost(self):
        """Find and evaluate the complementarity cost"""
        for cost in self.prog.GetAllCosts():
            if cost.evaluator().get_description() == self.cstr.name + "SlackCost":
                break
        dvals = [1.]*len(cost.variables())
        return cost.evaluator().Eval(dvals)

    def test_increasing_cost_weight(self):
        """Test that we can change the complementarity cost weight"""
        self.cstr.cost_weight = 1
        val = self.eval_complementarity_cost()
        self.cstr.cost_weight = 10
        new_val = self.eval_complementarity_cost()
        self.assertEqual(new_val, 10*val, msg = "Failed to change cost linearly with cost weight")

    def test_slack_variable(self):
        """Check that the final slack variable is close to 0"""
        result = self.solve_problem()
        slacks = self.cstr.var_slack
        slack_vals = result.GetSolution(slacks)
        np.testing.assert_allclose(slack_vals[-1:], np.zeros((1,)), err_msg = "Solved values for relaxation slacks is incorrect")

class CollocatedComplementarityCostRelaxedTest(CollocatedComplementarityBaseTestMixin, unittest.TestCase):
    def setUp(self):
        super(CollocatedComplementarityCostRelaxedTest, self).setUp()
        # Update expected problem parameters
        self.expected_num_constraints -= 1
        self.expected_num_costs += 1

    def add_complementarity_constraints(self):
        """Add collocated complementarity constraint to the program"""
        self.cstr = cp.CollocatedCostRelaxedComplementarity(self.fcn, self.x.shape[0], self.z.shape[0])
        self.cstr.addToProgram(self.prog, self.x, self.z)

    def eval_complementarity_cost(self):
        """Find and evaluate the complementarity cost"""
        for cost in self.prog.GetAllCosts():
            if cost.evaluator().get_description() == self.cstr.name + "_cost":
                break
        dvals = [1.]*len(cost.variables())
        return cost.evaluator().Eval(dvals)

    def test_increasing_cost_weight(self):
        """Test that we can change the complementarity cost weight"""
        self.cstr.cost_weight = 1
        val = self.eval_complementarity_cost()
        self.cstr.cost_weight = 10
        new_val = self.eval_complementarity_cost()
        self.assertEqual(new_val, 10*val, msg = "Failed to change cost linearly with cost weight")

if __name__ == '__main__':
    unittest.main()
