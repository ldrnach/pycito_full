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

class RelaxedLinearConstraintTest(unittest.TestCase):
    def setUp(self):
        # Set up the constraint and program
        A = np.array([[2, 0], [0, 3]])
        b = np.array([4, -9])
        self.lin_cstr = cstr.RelaxedLinearConstraint(A, b)
        self.x_expected = np.array([2, -3])
        # Set up the program
        self.prog = MathematicalProgram()
        self.x = self.prog.NewContinuousVariables(rows=2, name='x')
        self.lin_cstr.addToProgram(self.prog, self.x)

    def test_num_constraints(self):
        """Assert the correct number of constraints has been added"""
        self.assertEqual(len(self.prog.GetAllConstraints()), 3, msg='Unexpected number of constraints')
    
    def test_num_costs(self):
        """Assert the correct number of costs has been added"""
        self.assertEqual(len(self.prog.GetAllCosts()), 1, msg="Unexpected number of costs added")

    def test_solve(self):
        """Solve the program"""
        self.prog.SetInitialGuess(self.x, self.x_expected)
        self.prog.SetInitialGuess(self.lin_cstr.relax, np.zeros((1,)))
        result = Solve(self.prog)
        # Check that the problem is solved accurately
        self.assertTrue(result.is_success(), f'Failed to solve mathematical program with solver {result.get_solver_id().name()}')
        # Check the result
        np.testing.assert_allclose(result.GetSolution(self.x), self.x_expected, atol=1e-6, err_msg="Solution to the program is inaccurate")

    def test_cost_weight(self):
        """Test updating the cost weight"""
        testval = np.ones((1, ))
        cost = self.lin_cstr._cost.evaluator().Eval(testval)
        np.testing.assert_allclose(cost, np.ones((1,)), atol=1e-6, err_msg="Cost before changing weight inaccurate")
        self.lin_cstr.cost_weight=10
        cost = self.lin_cstr._cost.evaluator().Eval(testval)
        np.testing.assert_allclose(cost, 10*np.ones((1,)), atol=1e-6, err_msg="Cost after updating cost weight inaccurate")
        
class LinearImplicitDynamicsTest(unittest.TestCase):

    def setUp(self):
        """Setup a program with a linear constraint"""
        A = np.array([[2., 1.], [0., -3]])
        b = np.array([0., 6.])
        self.implicit_cstr = cstr.LinearImplicitDynamics(A, b)
        self.prog = MathematicalProgram()
        self.x = self.prog.NewContinuousVariables(rows = 2, name='x')
        self.x_expected = np.array([-1., 2.])

    def test_random(self):
        """Check that we can generate a random constraint"""
        in_dim = 3
        out_dim = 4
        rand_cstr = cstr.LinearImplicitDynamics.random(in_dim, out_dim)
        self.assertTrue(type(rand_cstr) is cstr.LinearImplicitDynamics, msg="LinearImplicitDynamics.random returns a constraint that is of the wrong type")
        self.assertEqual(rand_cstr.A.shape, (out_dim, in_dim), f"random constraint A matrix is the wrong shape")
        self.assertEqual(rand_cstr.b.shape, (out_dim, ), "random constraint b vector has the wrong shape")

    def test_add_constraint(self):
        """Test that adding the constraint adds 1 constraint to the problem"""
        self.assertEqual(len(self.prog.GetAllConstraints()), 0, f"Program has constraints before any are added")
        self.implicit_cstr.addToProgram(self.prog, self.x)
        self.assertEqual(len(self.prog.GetAllConstraints()), 1, f"Incorrect number of constraints added to program")

    def test_solve(self):
        """Check that we can solve the problem"""
        self.implicit_cstr.addToProgram(self.prog, self.x)
        self.prog.SetInitialGuess(self.x, self.x_expected)
        result = Solve(self.prog)
        np.testing.assert_allclose(result.GetSolution(self.x), self.x_expected, atol=1e-8, err_msg=f"Solution value inaccurate")

    def test_update_coefficients(self):
        """Check that we can update the constraint coefficients"""
        self.implicit_cstr.addToProgram(self.prog, self.x)
        Anew = 2*np.eye(2)
        bnew = np.array([4., -7.])
        self.implicit_cstr.updateCoefficients(Anew, bnew)
        evaluator = self.implicit_cstr._cstr.evaluator()
        np.testing.assert_allclose(self.implicit_cstr.A, Anew, atol=1e-8, err_msg="Calling updateCoefficients did not update the coeffcient matrix within the constriant object")
        np.testing.assert_allclose(self.implicit_cstr.b, bnew, atol=1e-8, err_msg='Calling updateCoefficients did not update the coefficient vector within the constraint object')
        np.testing.assert_allclose(evaluator.A(), Anew, atol=1e-8, err_msg="Calling updateCoefficients did not update the coefficient matrix in the constraint")
        np.testing.assert_allclose(evaluator.lower_bound(), bnew, atol=1e-8, err_msg="Calling updateCoefficients did not update the constraint lower bound")
        np.testing.assert_allclose(evaluator.upper_bound(), bnew, atol=1e-8, err_msg="Calling updateCoefficients did not update the constraint upper bound")


if __name__ == '__main__':
    unittest.main()