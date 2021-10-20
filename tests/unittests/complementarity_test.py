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
from trajopt import complementarity as cp
from pydrake.all import MathematicalProgram, Solve

def pass_through(x):
    return x

def example_constraint(z):
    x1, x2, y = np.split(z, 3)
    return y - (x1 - x2)**2

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
        cstr.slack = 1.
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