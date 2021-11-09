"""
unittests for trajopt.constriants

Luke Drnach

November 9, 2021
"""

import unittest
import numpy as np
from trajopt import constraints as cstr
from pydrake.all import MathematicalProgram, Solve

class CollocationTest(unittest.TestCase):
    def setUp(self):
        # Set up the program
        self.prog = MathematicalProgram()
        self.N = 2  # Number of variables
        self.K = 3  # Collocation order
        self.x = self.prog.NewContinuousVariables(rows = self.N, cols = self.K+1, name='x')
        self.x0 = self.prog.NewContinuousVariables(rows = self.N, cols = 1, name='x0')
        self.dx = self.prog.NewContinuousVariables(rows = self.N, cols = self.K+1, name = 'dx')
        self.h = self.prog.NewContinuousVariables(rows = 1, cols = 1, name='timestep')
        self.colloc = cstr.RadauCollocationConstraint(xdim = self.x.shape[0], order=self.K)
        self.colloc.addToProgram(self.prog, self.h, self.x, self.dx, self.x0)

    def solve_program(self):
        """Helper method to solve the program and return the result"""
        result = Solve(self.prog)
        self.assertTrue(result.is_success, msg="Collocation program solved unsuccessfully")
        return result

    def test_num_variables(self):
        """Check that we have the expected number of variables"""
        expected = self.N * (2*self.K + 3) + 1
        self.assertEqual(self.prog.decision_variables().size, expected, msg='Unexpected number of variables')

    def test_num_constraints(self):
        """Check that we've added the expected number of constraints to the program"""
        expected = 2*self.N #Expected vector constraints added to the program
        self.assertEqual(len(self.prog.GetAllConstraints()), expected, msg="Unexpected number of vector constraints")
        # Check the expected number of vector constraints
        expected = self.N * (self.K + 1)    # Expected number of scalar constraints
        total_out = 0
        for cs in self.prog.GetAllConstraints():
            dummy = [1.]*len(cs.variables())
            total_out += cs.evaluator().Eval(dummy).size
        self.assertEqual(total_out, expected, msg="Unexpected number of total scalar constraints")

    # def test_solution_verification(self):
    #     """Check that we can verify the solution"""
    #     # Set an initial guess to be the solution
    #     pass

    # def test_solution(self):
    #     """Tests that we can find the appropriate solution"""
    #     # Set an initial guess that is not the solution
    #     pass