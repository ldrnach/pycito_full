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

class TestLagrangeBasis(unittest.TestCase):

    def setUp(self):
        self._nodes = np.array([0., 1./3, 1.])
        self._values = self._nodes **3
        self.basis = [LagrangeBasis(self._nodes,n) for n in range(3)]
        
    def test_basis_weights(self):
        weights = np.asarray([base.weight for base in self.basis])
        expected = np.array([3, -9./2, 3./2])
        np.testing.assert_allclose(weights, expected, err_msg='Basis polynomial weights incorrect')

    def test_basis_eval(self):
        val = np.concatenate([base.eval(1./2) for base in self.basis])
        expected = np.array([-1./4, 9./8, 1./8])
        np.testing.assert_allclose(val, expected, err_msg="Evaluating basis polynomials inaccurate")
        
    def test_basis_eval_vector(self):
        times = np.array([1/2, 2/3])
        val = np.row_stack([base.eval(times) for base in self.basis]).transpose()
        expected = np.array([[-1/4, 9/8, 1/8], 
                             [-1/3,  1.,  1/3]])
        np.testing.assert_allclose(val, expected, err_msg = "Evaluating basis with a vector of times is incorrect")

    def test_basis_derivative(self):
        val = np.concatenate([base.derivative(1./2) for base in self.basis])
        expected = np.array([-1, 0, 1])
        np.testing.assert_allclose(val, expected, err_msg="Evaluating derivatives of basis polynomials incorrect")

    def test_basis_derivative_vector(self):
        times = np.array([1/2, 2/3])
        val = np.row_stack([base.derivative(times) for base in self.basis]).transpose()
        expected = np.array([[-1, 0, 1],
                             [0, -3/2, 3/2]])
        np.testing.assert_allclose(val, expected, atol = 1e-7, err_msg="Evaluating derivatives of basis polynomials with a vector of times is incorrect")

class TestLagrangeInterpolant(unittest.TestCase):
    def setUp(self):
        self._nodes = np.array([0., 1./3, 1.])
        self._values = self._nodes **3
        self.poly = LagrangeInterpolant(self._nodes, self._values)

    def test_interpolation_scalar(self):
        val = self.poly.eval(1./2)
        expected = np.array([1./6])
        np.testing.assert_allclose(val, expected, err_msg="Evaluating Interpolating polynomial incorrect")

    def test_interpolation_vector(self):
        times = np.array([1/2, 2/3])
        val = self.poly.eval(times)
        expected = np.array([1/6, 10/27])
        np.testing.assert_allclose(val, expected, err_msg='Evalutaing interpolant at multiple times is inaccurate')

    def test_interpolation_derivative_scalar(self):
        val = self.poly.derivative(1./2)
        expected = np.asarray([1.])
        np.testing.assert_allclose(val, expected, err_msg="Evaluating interpolant derivative incorrect")

    def test_interpolation_derivative_vector(self):
        times = np.asarray([1/2, 2/3])
        val = self.poly.derivative(times)
        expected = np.array([1., 13/9])
        np.testing.assert_allclose(val, expected, err_msg="Evaluting interpolant derivative at multiple times is inaccurate")

    def test_differentiation_matrix(self):
        Dval = self.poly.differentiation_matrix
        expected = np.array([[-4, 9./2, -1./2],
                            [-2., 3./2, 1./2],
                            [2., -9./2, 5./2]])
        np.testing.assert_allclose(Dval, expected, err_msg="Differentiation matrix incorrect")
    #TODO: Interpolation returns an extra dimension when recovering original points
    def test_interpolation_recovery(self):
        node_val = np.concatenate([self.poly.eval(val) for val in self._nodes])
        np.testing.assert_allclose(node_val, self._values, err_msg="Failed to recover all node values sequentially")   

    def test_interpolation_recovery_vector(self):
        node_val = self.poly.eval(self._nodes)
        np.testing.assert_allclose(node_val, self._values, err_msg="Failed to recover all node values simulateously")

class TestLagrangeVectorInterpolation(unittest.TestCase):
    def setUp(self):
        self._nodes = np.array([0., 1./3, 1.])
        self._values = np.zeros((2,3))
        self._values[0,:] = self._nodes **3
        self._values[1,:] = np.cos(np.pi*self._nodes)
        self.poly = LagrangeInterpolant(self._nodes, self._values)
        self.test_times = np.array([1/2, 2/3])

    def test_interpolation_scalar(self):
        time = self.test_times[0]
        val = self.poly.eval(time)
        expected = np.array([[1/6], [3/16]])
        np.testing.assert_allclose(val, expected, err_msg='Evaluating a vector interpolant at a single time incorrect')

    def test_interpolation_vector(self):
        val = self.poly.eval(self.test_times)
        expected = np.array([[1/6, 10/27], [3/16, -1/6]])
        np.testing.assert_allclose(val, expected, err_msg='Evaluating vector interpolant at multiple times is incorrect')

    def test_derivative_scalar(self):
        val = self.poly.derivative(self.test_times[0])
        expected = np.array([[1. ], [-2]])
        np.testing.assert_allclose(val, expected, err_msg='Evaluating vector derivative at a single time incorrect')

    def test_derivative_vector(self):
        val = self.poly.derivative(self.test_times)
        expected = np.array([[1., 13/9], [-2, -9/4]])
        np.testing.assert_allclose(val, expected, err_msg="Evaluting vector derivative at multiple times incorrect")

    def test_interpolation_recovery(self):
        node_val = self.poly.eval(self._nodes)
        np.testing.assert_allclose(node_val, self._values, err_msg="Interpolant does not return original values")   