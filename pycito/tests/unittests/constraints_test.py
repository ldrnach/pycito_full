"""
unittests for trajopt.constriants

Luke Drnach

November 9, 2021
"""

import unittest
import numpy as np
from pycito.trajopt import constraints as cstr
from pydrake.all import MathematicalProgram, Solve

def polynomial_example(t):
    """Cubic polynomial example for testing collocation"""
    # Initialize
    x = np.zeros((2, t.size))
    dx = np.zeros((2, t.size))
    # Create two polynomials (at most cubics)
    x[0,:] = 3*t**3 - t**2 + 2*t + 1
    x[1,:] = -2*t**2 + 3*t - 2
    # Also return their derivatives
    dx[0,:] = 9*t**2 - 2 * t + 2
    dx[1,:] = -4*t + 3
    return x, dx
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
        # Calculate test values
        val0, _ = polynomial_example(np.zeros((1,)))
        vals, derivs = polynomial_example(self.colloc.nodes)
        self.test_vals = (val0, vals, derivs)

    def solve_program(self):
        """Helper method to solve the program and return the result"""
        result = Solve(self.prog)
        self.assertTrue(result.is_success, msg="Collocation program solved unsuccessfully")
        return result

    def test_num_variables(self):
        """Check that we have the expected number of variables"""
        expected = self.N * (2*self.K + 3) + 1
        self.assertEqual(self.prog.decision_variables().size, expected, msg='Unexpected number of variables')

    def test_polynomial_values(self):
        """Check that the polynomial values and derivatives match"""
        v0, vals, derivs = self.test_vals
        self.colloc.values = vals
        pderivs = self.colloc.derivative(self.colloc.nodes)
        np.testing.assert_allclose(derivs, pderivs, rtol=0, atol=1e-6, err_msg = 'Polynomial derivatives do not match collocation polynomial')

    def test_num_constraints(self):
        """Check that we've added the expected number of constraints to the program"""
        expected = 2*self.N #Expected vector constraints added to the program
        self.assertEqual(len(self.prog.GetAllConstraints()), expected, msg="Unexpected number of vector constraints")
        # Check the expected number of vector constraints
        expected = self.N * (self.K + 1)    # Expected number of scalar constraints
        total_out = 0
        for cs in self.prog.GetAllConstraints():
            total_out += cs.evaluator().num_constraints()
        self.assertEqual(total_out, expected, msg="Unexpected number of total scalar constraints")

    def test_solve_states(self):
        """Given the derivatives, solve for the states"""
        # Calculate the values and derivatives of the target polynomial
        val0, vals, derivs = self.test_vals
        # Fix timestep, derivatives, and initial condition
        self.prog.AddBoundingBoxConstraint(np.ones(self.h.shape), np.ones(self.h.shape), self.h)
        self.prog.AddBoundingBoxConstraint(derivs.flatten(), derivs.flatten(), self.dx.flatten())
        self.prog.AddBoundingBoxConstraint(val0.flatten(), val0.flatten(), self.x0.flatten())
        # Initialize the state values
        self.prog.SetInitialGuess(self.x, np.ones(self.x.shape))
        # Solve the program
        result = self.solve_program()
        soln = result.GetSolution(self.x)
        np.testing.assert_allclose(soln, vals, rtol = 0, atol = 1e-6, err_msg='Solved polynomial values are incorrect')

    def test_solve_derivatives(self):
        """Given the states, solve for the derivatives"""
        # Calculate values and derivatives of target polynomial
        val0, vals, derivs = self.test_vals
        # Fix timestep and values
        self.prog.AddBoundingBoxConstraint(np.ones(self.h.shape), np.ones(self.h.shape), self.h)
        self.prog.AddBoundingBoxConstraint(vals.flatten(), vals.flatten(), self.x.flatten())
        # Fix last derivative - it's a free variable anyway
        self.prog.AddBoundingBoxConstraint(derivs[:, -1].flatten(), derivs[:, -1].flatten(), self.dx[:, -1].flatten())
        # Initialize the derivatives
        self.prog.SetInitialGuess(self.dx[:, :-1], np.ones(self.dx[:, :-1].shape))
        # Solve the problem
        result = self.solve_program()
        soln_dx = result.GetSolution(self.dx)
        soln_x0 = result.GetSolution(self.x0)
        # Check the derivatives
        np.testing.assert_allclose(soln_dx, derivs, rtol=0, atol=1e-6, err_msg='Solved derivatives are incorrect')
        # Check the initial condition
        np.testing.assert_allclose(soln_x0, val0.flatten(), rtol=0, atol=1e-6, err_msg='Solved initial condition is incorrect')




if __name__ == '__main__':
    unittest.main()