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
from numpy.testing._private.utils import assert_equal
from trajopt import complementarity as cp
from pydrake.all import MathematicalProgram, Solve

def pass_through(x):
    return x

def example_constraint(z):
    x1, x2, y = np.split(z, 3)
    return y - (x1 - x2)**2

def test_constraint(x):
    return x - 1

class ComplementarityTest(unittest.TestCase):
    def setUp(self):
        # Create the program and associated variables
        self.prog = MathematicalProgram()
        self.x = self.prog.NewContinuousVariables(rows=2, cols=1, name='x')
        self.y = self.prog.NewContinuousVariables(rows=1, cols=1, name='y')
        # Add the cost and constraint
        self.prog.AddLinearCost(a = np.ones((2,)), b = 0., vars = self.x)
        self.prog.AddConstraint(example_constraint, lb=np.ones((1,)), ub=np.ones((1,)), vars=np.concatenate([self.x, self.y]))
        # Set the initial guess for the solution to all ones
        self.prog.SetInitialGuess(self.x, np.ones((2,)))
        self.prog.SetInitialGuess(self.y, np.ones((1,)))

    def solve_problem(self, cstr):
        """ Helper function to solve the complementarity problem """
        # Check that the problem solves successfully
        result = Solve(self.prog)
        self.assertTrue(result.is_success(), msg=f"Failed to solve complementarity problem with {type(cstr).__name__}")
        return result

    def check_result(self, result, cstr):
        """ Check that the results are reasonably accurate"""
        x_ = result.GetSolution(self.x)
        y_ = result.GetSolution(self.y)
        np.testing.assert_allclose(x_, np.zeros((2,)), atol= 1e-3, err_msg=f"Results are not close to the expected solution using {type(cstr).__name__}")
        np.testing.assert_allclose(y_, np.ones((1,)), atol=1e-3, err_msg =f"Results are not close to the expected solution using {type(cstr).__name__}")

    def get_constraint_upper_bound(self):
        """ Get the upper bound for the complementarity constraint"""
        for cstr in self.prog.GetAllConstraints():
            if cstr.evaluator().get_description() == "pass_through":
                return cstr.evaluator().upper_bound()

    def check_constant_slack(self, cstr):
        """ Test snippet for checking if setting a constant slack changes the upper bounds"""
        ub_1 = self.get_constraint_upper_bound()
        cstr.const_slack = 1.
        ub_2 = self.get_constraint_upper_bound()
        self.assertTrue(np.any(np.not_equal(ub_1, ub_2)), msg=f"Changing constant slack does not change the upper bound for {type(cstr).__name__}")

    def check_setting_cost(self, cstr, result):
        """ Test snippet for checking if setting a cost weight works"""
        cstr.cost_weight = 1.
        cost1 = self.eval_complementarity_cost(result)
        cstr.cost_weight = 10.
        cost2 = self.eval_complementarity_cost(result)
        self.assertAlmostEqual(cost1*10, cost2, delta=1e-7, msg=f"Setting cost_weight does not change cost for {type(cstr).__name__}")

    def eval_complementarity_cost(self, result):
        """ Evaluate the cost associated with the complementarity constraint"""
        costs = self.prog.GetAllCosts()
        for cost in costs:
            if cost.evaluator().get_description() in ["pass_throughSlackCost", "pass_throughCost"]:
                dvars = cost.variables()
                dvals = result.GetSolution(dvars)
                break
        return cost.evaluator().Eval(dvals) 

    def test_nonlinear_constant_slack(self):
        """Test NonlinearComplementarityConstantSlack """
        # Test with zero slack
        cstr = cp.NonlinearConstantSlackComplementarity(pass_through, xdim=1, zdim=1)
        cstr.addToProgram(self.prog, xvars=self.x[0], zvars=self.x[1])
        result = self.solve_problem(cstr)
        # Check the answer
        self.check_result(result, cstr)
        # Check changing the constant slack
        self.check_constant_slack(cstr)

    def test_nonlinear_variable_slack(self):
        """ Test NonlinearComplementarityVariableSlack constraint"""
        # Test with cost-weight 1
        cstr = cp.NonlinearVariableSlackComplementarity(pass_through, xdim=1, zdim=1)
        cstr.addToProgram(self.prog, xvars=self.x[0], zvars=self.x[1])
        result = self.solve_problem(cstr)
        # Check the answer
        self.check_result(result, cstr)
        # Check changing the cost weight
        self.check_setting_cost(cstr, result)

    def test_nonlinear_cost(self):
        """ Test CostRelaxedNonlinearComplementarity"""
        cstr = cp.CostRelaxedNonlinearComplementarity(pass_through, xdim=1, zdim=1)
        cstr.addToProgram(self.prog, xvars=self.x[0], zvars=self.x[1])
        result = self.solve_problem(cstr)
        # Check the result
        self.check_result(result, cstr)
        # Check changing the cost
        self.check_setting_cost(cstr, result)

    def test_linear_constant_slack(self):
        """ Test LinearEqualityConstantSlackComplementarity """
        cstr = cp.LinearEqualityConstantSlackComplementarity(pass_through, xdim=1, zdim=1)
        cstr.addToProgram(self.prog, xvars=self.x[0], zvars=self.x[1])
        result = self.solve_problem(cstr)
        # Check the result
        self.check_result(result, cstr)
        # Check changing the constant slack
        self.check_constant_slack(cstr)
        
    def test_linear_variable_slack(self):
        """ Test LinearEqualityVariableSlackComplementarity """
        cstr = cp.LinearEqualityVariableSlackComplementarity(pass_through, xdim=1, zdim=1)
        cstr.addToProgram(self.prog, xvars=self.x[0], zvars=self.x[1])
        result = self.solve_problem(cstr)
        # Check the result
        self.check_result(result, cstr)
        # Check changing the cost
        self.check_setting_cost(cstr, result)

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