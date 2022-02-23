"""
unittests for pycito.systems.kernels.py

Luke Drnach
February 23, 2022
"""
#TODO: Handle cases where input to kernel is (N, 1) instead of (N, )
import unittest
import numpy as np
import pydrake.autodiffutils as ad

import pycito.systems.kernels as kernels

class RBFKernelTest(unittest.TestCase):
    def setUp(self):
        self.kernel = kernels.RBFKernel()
        self.x = np.array([[1, 0], [0, 1], [-1, 1]])
        self.y = np.array([[2, -1], [-3, 2]])
        self.Kxx_expected = np.exp(np.array([[0,    -1, -5/2], 
                                            [-1,     0, -1/2],
                                            [-5/2, -1/2, 0]]))
        self.Kxy_expected = np.exp(np.array([[-1,   -10], 
                                            [-4,    -5],
                                            [-13/2, -5/2]]))

    def test_eval(self):
        # test that we can evaluate the kernel for single inputs
        k = self.kernel.eval(self.x[0,:], self.y[0,:])
        np.testing.assert_allclose(k, self.Kxy_expected[0,0], atol=1e-6, err_msg=f"Kernel evaluation inaccurate for single inputs")
        # test normalization
        kn = self.kernel.eval(self.y[0, :], self.y[0, :])
        np.testing.assert_allclose(kn, np.ones((1,)), atol=1e-6, err_msg=f"Kernel is not normalized")

    def test_gradient(self):
        # Test that we can evaluate the kernel gradient on single inputs
        x_ad = np.squeeze(ad.InitializeAutoDiff(self.x[2,:]))
        k_ad = self.kernel.eval(x_ad, self.y[0,:])
        dk_ad = k_ad.derivatives()
        # Evaluate the gradient directly
        dk = self.kernel.gradient(self.x[2,:], self.y[0,:])
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calcuated by autodiff")

    def test_call_symmetric(self):
        # Test that calling the kernel alone results in a symmetric matrix
        Kxx = self.kernel(self.x)
        np.testing.assert_allclose(Kxx, self.Kxx_expected, atol=1e-6, err_msg=f"Calling kernel on a single dataset results in inaccurate kernel matrix")

    def test_call_asymmetric(self):
        # Test that calling the kernel on two datasets results in an asymmetric matrix
        Kxy = self.kernel(self.x, self.y)
        np.testing.assert_allclose(Kxy, self.Kxy_expected, atol=1e-6, err_msg=f"Calling kernel on two datasets results in inaccurate asymmetric kernel matrix")

class PsuedoHuberKernelTest(unittest.TestCase):
    def setUp(self):
        self.kernel = kernels.PseudoHuberKernel()
        self.x = np.array([[1, 0], [0, 1], [-1, 1]])
        self.y = np.array([[2, -1], [-3, 2]])
        dxy = np.array([[3, 21], [9, 11], [14, 6]])
        dxx = np.array([[1, 3, 6], [3, 1, 2], [6, 2, 1]])
        self.Kxy_expected = np.exp(1 - np.sqrt(dxy))
        self.Kxx_expected = np.exp(1 - np.sqrt(dxx))

    def test_eval(self):
        # test that we can evaluate the kernel for single inputs
        k = self.kernel.eval(self.x[0,:], self.y[0,:])
        np.testing.assert_allclose(k, self.Kxy_expected[0,0], atol=1e-6, err_msg=f"Kernel evaluation inaccurate for single inputs")
        # test normalization
        kn = self.kernel.eval(self.y[0, :], self.y[0, :])
        np.testing.assert_allclose(kn, np.ones((1,)), atol=1e-6, err_msg=f"Kernel is not normalized")

    def test_gradient(self):
        # Test that we can evaluate the kernel gradient on single inputs
        x_ad = np.squeeze(ad.InitializeAutoDiff(self.x[2,:]))
        k_ad = self.kernel.eval(x_ad, self.y[0,:])
        dk_ad = k_ad.derivatives()
        # Evaluate the gradient directly
        dk = self.kernel.gradient(self.x[2,:], self.y[0,:])
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calcuated by autodiff")

    def test_call_symmetric(self):
        # Test that calling the kernel alone results in a symmetric matrix
        Kxx = self.kernel(self.x)
        np.testing.assert_allclose(Kxx, self.Kxx_expected, atol=1e-6, err_msg=f"Calling kernel on a single dataset results in inaccurate kernel matrix")

    def test_call_asymmetric(self):
        # Test that calling the kernel on two datasets results in an asymmetric matrix
        Kxy = self.kernel(self.x, self.y)
        np.testing.assert_allclose(Kxy, self.Kxy_expected, atol=1e-6, err_msg=f"Calling kernel on two datasets results in inaccurate asymmetric kernel matrix")

if __name__ == '__main__':
    unittest.main()