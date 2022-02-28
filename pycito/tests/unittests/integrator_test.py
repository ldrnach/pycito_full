"""
Unittest for systems.integrators.py
NOTE: Since we're using the block system here, the dynamics are a linear, time-invariant complementarity system. So all integrators should yield the same results for velocity, but different results for the position

Luke Drnach
February 28, 2022
"""

import numpy as np
import unittest
from pycito.systems.integrators import ContactDynamicsIntegrator
from pycito.systems.block.block import Block

class IntegratorTest(unittest.TestCase):
    def setUp(self):
        # Set up the example problem
        self.plant = Block()
        self.plant.terrain.friction = 0.5
        self.plant.Finalize()
        # Initial state:
        self.x0 = np.array([0, 0.5, 0., 0.]).T
        self.u = np.array([14.905])
        self.dt = np.array([0.1])
        # Expected state 
        self.x_implicit = np.array([0.1, 0.5, 1.0, 0.0]).T
        self.x_explicit = np.array([0.0, 0.5, 1.0, 0.0]).T
        self.x_midpoint = np.array([0.05, 0.5, 1.0, 0.]).T
        # Expected forces
        self.f = np.array([9.81, 0.0, 0.0, 4.905, 0.0]).T

    def test_implicit_euler(self):
        """Test ImplicitEulerIntegrator"""
        integrator = ContactDynamicsIntegrator.ImplicitEulerIntegrator(self.plant)
        x, f, status = integrator.integrate(self.dt, self.x0, self.u)
        self.assertTrue(status, msg="ImplicitEulerIntegrator failed to   advance the state")
        # Explicit euler final stat
        np.testing.assert_allclose(x, self.x_implicit, atol=1e-6, err_msg = f"ImplicitEulerIntegrator returned an inaccurate solution for the state")
        np.testing.assert_allclose(f, self.f, atol=1e-6, err_msg=f"ImplicitEulerIntegrator returned inaccurate solution for reaction force ")

    def test_semi_implicit(self):
        """Test the SemiImplicit Euler Integrator"""
        integrator = ContactDynamicsIntegrator.SemiImplicitEulerIntegrator(self.plant)
        x, f, status = integrator.integrate(self.dt, self.x0, self.u)
        self.assertTrue(status, msg="Semi-ImplicitEulerIntegrator failed to   advance the state")
        np.testing.assert_allclose(x, self.x_implicit, atol=1e-6, err_msg = f"Semi-IplicitEulerIntegrator returned an inaccurate solution for the state")
        np.testing.assert_allclose(f, self.f, atol=1e-6, err_msg=f"Semi-ImplicitEulerIntegrator returned inaccurate solution for reaction force ")

    def test_implicit_midpoint(self):
        """Test the Midpoint Euler Integrator"""
        integrator = ContactDynamicsIntegrator.ImplicitMidpointIntegrator(self.plant)
        x, f, status = integrator.integrate(self.dt, self.x0, self.u)
        self.assertTrue(status, msg="ImplicitMidpointIntegrator failed to advance the state")
        np.testing.assert_allclose(x, self.x_midpoint, atol=1e-6, err_msg = f"ImplicitMidpointIntegrator returned an inaccurate solution for the state")
        np.testing.assert_allclose(f, self.f, atol=1e-6, err_msg=f"ImplicitMidpointIntegrator returned inaccurate solution for reaction force ")

if __name__ == '__main__':
    unittest.main()