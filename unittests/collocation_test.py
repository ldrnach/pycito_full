"""
unittests for trajopt.collocation

Tests getting the weights, derivatives, and interpolation values from the Lagrange Basis Polynomials as well as the Lagrange Interpolating Polynomial

Luke Drnach
September 28, 2021
"""

#TODO: Implement unittests for vector function interpolation
#TODO: Implement unittests for Radau Collocation
#TODO: Document the unittests

import numpy as np
import unittest
from trajopt.collocation import LagrangeBasis, LagrangeInterpolant

class TestLagrangeInterpolation(unittest.TestCase):

    def setUp(self):
        self._nodes = np.array([0., 1./3, 1.])
        self._values = self._nodes **3
        self.basis = [LagrangeBasis(self._nodes,n) for n in range(3)]
        self.poly = LagrangeInterpolant(self._nodes, self._values)

    def test_basis_weights(self):
        weights = np.asarray([base.weight for base in self.basis])
        expected = np.array([3, -9./2, 3./2])
        np.testing.assert_allclose(weights, expected, err_msg='Basis polynomial weights incorrect')

    def test_basis_eval(self):
        val = np.concatenate([base.eval(1./2) for base in self.basis])
        expected = np.array([-1./4, 9./8, 1./8])
        np.testing.assert_allclose(val, expected, err_msg="Evaluating basis polynomials inaccurate")
        
    def test_basis_derivative(self):
        val = np.concatenate([base.derivative(1./2) for base in self.basis])
        expected = np.array([-1, 0, 1])
        np.testing.assert_allclose(val, expected, err_msg="Evaluating derivatives of basis polynomials incorrect")

    def test_interpolation(self):
        val = self.poly.eval(1./2)
        expected = np.array([1./6])
        np.testing.assert_allclose(val, expected, err_msg="Evaluating Interpolating polynomial incorrect")

    def test_interpolation_derivative(self):
        val = self.poly.derivative(1./2)
        expected = np.asarray([[1.]])
        np.testing.assert_allclose(val, expected, err_msg="Evaluating interpolant derivative incorrect")

    def test_differentiation_matrix(self):
        Dval = self.poly.differentiation_matrix
        expected = np.array([[-4, 9./2, -1./2],
                            [-2., 3./2, 1./2],
                            [2., -9./2, 5./2]])
        np.testing.assert_allclose(Dval, expected, err_msg="Differentiation matrix incorrect")

    def test_interpolation_recovery(self):
        nodeval = np.concatenate([self.poly.eval(val) for val in self._nodes])
        np.testing.assert_allclose(nodeval, np.expand_dims(self._values, axis=1), err_msg="Interpolant does not return original values")   