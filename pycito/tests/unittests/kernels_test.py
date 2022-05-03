"""
unittests for pycito.systems.kernels.py

Luke Drnach
February 23, 2022
"""
import unittest
import numpy as np
import pydrake.autodiffutils as ad

import pycito.systems.kernels as kernels

class RBFKernelTest(unittest.TestCase):
    def setUp(self):
        self.kernel = kernels.RBFKernel()
        self.x = np.array([[1, 0], [0, 1], [-1, 1]]).T
        self.y = np.array([[2, -1], [-3, 2]]).T
        self.Kxx_expected = np.exp(np.array([[0,    -1, -5/2], 
                                            [-1,     0, -1/2],
                                            [-5/2, -1/2, 0]]))
        self.Kxy_expected = np.exp(np.array([[-1,   -10], 
                                            [-4,    -5],
                                            [-13/2, -5/2]]))

    def test_eval(self):
        """Test evaluation of the kernel for float inputs"""
        # test that we can evaluate the kernel for single inputs
        k = self.kernel.eval(self.x[:, 0], self.y[:, 0])
        np.testing.assert_allclose(k, self.Kxy_expected[:1, :1], atol=1e-6, err_msg=f"Kernel evaluation inaccurate for single inputs")
        # test normalization
        kn = self.kernel.eval(self.y[:, 0], self.y[:, 0])
        np.testing.assert_allclose(kn, np.ones((1,1)), atol=1e-6, err_msg=f"Kernel is not normalized")
        # Test on a single x vector and multiple y vectors
        k = self.kernel.eval(self.x[:, 0], self.y)
        np.testing.assert_allclose(k, self.Kxy_expected[:1, :], atol=1e-6, err_msg="Kernel evaluation fails for a single first input with multiple second inputs")
        # Test on a multiple x vectors and a single y vector
        k = self.kernel.eval(self.x, self.y[:, 0])
        np.testing.assert_allclose(k, self.Kxy_expected[:, :1], atol=1e-6, err_msg=f"Kernel evaluation fails for multiple first input vectors, but a single second input vector")

    def test_call_symmetric(self):
        # Test that calling the kernel alone results in a symmetric matrix
        Kxx = self.kernel(self.x)
        np.testing.assert_allclose(Kxx, self.Kxx_expected, atol=1e-6, err_msg=f"Calling kernel on a single dataset results in inaccurate kernel matrix")

    def test_gradient(self):
        """Test evaluation of the kernel gradients"""
        x_ad = np.squeeze(ad.InitializeAutoDiff(self.x[:, 2]))
        # Test evaluating the kernel gradient with a single y-sample
        k_ad = self.kernel.eval(x_ad, self.y[:, 0])
        dk_ad = ad.ExtractGradient(k_ad)
        dk = self.kernel.gradient(self.x[:, 2], self.y[:, 0])
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calculated by autodiff for single y-vector")
        # Test that we can evaluate the kernel gradient with multiple y-samples
        k_ad = self.kernel.eval(x_ad, self.y)
        dk_ad = ad.ExtractGradient(k_ad)
        # Evaluate the gradient directly
        dk = self.kernel.gradient(self.x[:, 2], self.y)
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calcuated by autodiff for multiple y-vectors")

    def test_eval_autodiff(self):
        """ Test evaluating the kernel when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single sample y-point
        Kxy_ad = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1, :1], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is vector")
        # Test with multiple input sample y-points
        Kxy_ad = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1,:], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is array")
    
    def test_gradient_autodiff(self):
        """Test evaluating the kernel gradient when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single y-point
        dKxy_ad  =self.kernel.gradient(x_ad, self.y[:, 0])
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is a single vector")
        # Test with multiple y-points
        dKxy_ad = self.kernel.gradient(x_ad, self.y)
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is multiple vectors")

class PsuedoHuberKernelTest(unittest.TestCase):
    def setUp(self):
        self.kernel = kernels.PseudoHuberKernel()
        self.x = np.array([[1, 0], [0, 1], [-1, 1]]).T
        self.y = np.array([[2, -1], [-3, 2]]).T
        dxy = np.array([[3, 21], [9, 11], [14, 6]])
        dxx = np.array([[1, 3, 6], [3, 1, 2], [6, 2, 1]])
        self.Kxy_expected = np.exp(1 - np.sqrt(dxy))
        self.Kxx_expected = np.exp(1 - np.sqrt(dxx))

    def test_eval(self):
        """Test evaluation of the kernel for float inputs"""
        # test that we can evaluate the kernel for single inputs
        k = self.kernel.eval(self.x[:, 0], self.y[:, 0])
        np.testing.assert_allclose(k, self.Kxy_expected[:1, :1], atol=1e-6, err_msg=f"Kernel evaluation inaccurate for single inputs")
        # test normalization
        kn = self.kernel.eval(self.y[:, 0], self.y[:, 0])
        np.testing.assert_allclose(kn, np.ones((1,1)), atol=1e-6, err_msg=f"Kernel is not normalized")
        # Test on a single x vector and multiple y vectors
        k = self.kernel.eval(self.x[:, 0], self.y)
        np.testing.assert_allclose(k, self.Kxy_expected[:1, :], atol=1e-6, err_msg="Kernel evaluation fails for a single first input with multiple second inputs")
        # Test on a multiple x vectors and a single y vector
        k = self.kernel.eval(self.x, self.y[:, 0])
        np.testing.assert_allclose(k, self.Kxy_expected[:, :1], atol=1e-6, err_msg=f"Kernel evaluation fails for multiple first input vectors, but a single second input vector")

    def test_call_symmetric(self):
        # Test that calling the kernel alone results in a symmetric matrix
        Kxx = self.kernel(self.x)
        np.testing.assert_allclose(Kxx, self.Kxx_expected, atol=1e-6, err_msg=f"Calling kernel on a single dataset results in inaccurate kernel matrix")

    def test_call_asymmetric(self):
        # Test that calling the kernel on two datasets results in an asymmetric matrix
        Kxy = self.kernel(self.x, self.y)
        np.testing.assert_allclose(Kxy, self.Kxy_expected, atol=1e-6, err_msg=f"Calling kernel on two datasets results in inaccurate asymmetric kernel matrix")

    def test_gradient(self):
        """Test evaluation of the kernel gradients"""
        x_ad = np.squeeze(ad.InitializeAutoDiff(self.x[:, 2]))
        # Test evaluating the kernel gradient with a single y-sample
        k_ad = self.kernel.eval(x_ad, self.y[:, 0])
        dk_ad = ad.ExtractGradient(k_ad)
        dk = self.kernel.gradient(self.x[:, 2], self.y[:, 0])
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calculated by autodiff for single y-vector")
        # Test that we can evaluate the kernel gradient with multiple y-samples
        k_ad = self.kernel.eval(x_ad, self.y)
        dk_ad = ad.ExtractGradient(k_ad)
        # Evaluate the gradient directly
        dk = self.kernel.gradient(self.x[:, 2], self.y)
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calcuated by autodiff for multiple y-vectors")

    def test_eval_autodiff(self):
        """ Test evaluating the kernel when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single sample y-point
        Kxy_ad = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1, :1], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is vector")
        # Test with multiple input sample y-points
        Kxy_ad = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1,:], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is array")

    def test_gradient_autodiff(self):
        """Test evaluating the kernel gradient when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single y-point
        dKxy_ad  =self.kernel.gradient(x_ad, self.y[:, 0])
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is a single vector")
        # Test with multiple y-points
        dKxy_ad = self.kernel.gradient(x_ad, self.y)
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is multiple vectors")

class LinearKernelTest(unittest.TestCase):
    """Unittests for the LinearKernel Class"""
    def setUp(self):
        W = np.eye(2)
        offset = np.ones((1,))
        self.kernel = kernels.LinearKernel(weights = W, offset = offset)
        self.x = np.array([[1., 0.], [0., 1.], [-1., 1.]]).T
        self.y = np.array([[2., -1.], [-3., 2.]]).T
        self.Kxy_expected = np.array([[3., 0., -2.],[-2., 3., 6.]]).T
        self.Kxx_expected = np.array([[2., 1., 0.], [1., 2., 2.], [0., 2., 3.]]).T

    def test_eval(self):
        """Check the evaluation of the kernel function"""
        Kxy = self.kernel.eval(self.x, self.y)
        np.testing.assert_allclose(Kxy, self.Kxy_expected, atol=1e-12, err_msg=f"Asymmetric Kernel evaluation produces incorrect kernel matrix")
        Kxx = self.kernel.eval(self.x, self.x)
        np.testing.assert_allclose(Kxx, self.Kxx_expected, atol=1e-12, err_msg=f"Symmetric Kernel evaluation produces incorrect kernel matrix")

    def test_gradient(self):
        """Test evaluation of the kernel gradients"""
        x_ad = np.squeeze(ad.InitializeAutoDiff(self.x[:, 2]))
        # Test evaluating the kernel gradient with a single y-sample
        k_ad = self.kernel.eval(x_ad, self.y[:, 0])
        dk_ad = ad.ExtractGradient(k_ad)
        dk = self.kernel.gradient(self.x[:, 2], self.y[:, 0])
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calculated by autodiff for single y-vector")
        # Test that we can evaluate the kernel gradient with multiple y-samples
        k_ad = self.kernel.eval(x_ad, self.y)
        dk_ad = ad.ExtractGradient(k_ad)
        # Evaluate the gradient directly
        dk = self.kernel.gradient(self.x[:, 2], self.y)
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calcuated by autodiff for multiple y-vectors")

    def test_eval_autodiff(self):
        """ Test evaluating the kernel when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single sample y-point
        Kxy_ad = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1, :1], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is vector")
        # Test with multiple input sample y-points
        Kxy_ad = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1,:], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is array")

    def test_gradient_autodiff(self):
        """Test evaluating the kernel gradient when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single y-point
        dKxy_ad  =self.kernel.gradient(x_ad, self.y[:, 0])
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is a single vector")
        # Test with multiple y-points
        dKxy_ad = self.kernel.gradient(x_ad, self.y)
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is multiple vectors")
    
class HyperbolicKernelTest(unittest.TestCase):
    """Unittests for the HyperbolicTangentKernel Class"""
    def setUp(self):
        W = np.eye(2)
        offset = np.ones((1,))
        self.kernel = kernels.HyperbolicTangentKernel(weights=W, offset=offset)
        self.x = np.array([[1, 0], [0, 1], [-1, 1]]).T
        self.y = np.array([[2, -1], [-3, 2]]).T
        self.Kxy_expected = np.tanh(np.array([[3., 0., -2.],[-2., 3., 6.]])).T
        self.Kxx_expected = np.tanh(np.array([[2., 1., 0.], [1., 2., 2.], [0., 2., 3.]])).T

    def test_eval(self):
        """Check the evaluation of the kernel function"""
        Kxy = self.kernel.eval(self.x, self.y)
        np.testing.assert_allclose(Kxy, self.Kxy_expected, atol=1e-12, err_msg=f"Asymmetric Kernel evaluation produces incorrect kernel matrix")
        Kxx = self.kernel.eval(self.x, self.x)
        np.testing.assert_allclose(Kxx, self.Kxx_expected, atol=1e-12, err_msg=f"Symmetric Kernel evaluation produces incorrect kernel matrix")

    def test_gradient(self):
        """Test evaluation of the kernel gradients"""
        x_ad = np.squeeze(ad.InitializeAutoDiff(self.x[:, 2]))
        # Test evaluating the kernel gradient with a single y-sample
        k_ad = self.kernel.eval(x_ad, self.y[:, 0])
        dk_ad = ad.ExtractGradient(k_ad)
        dk = self.kernel.gradient(self.x[:, 2], self.y[:, 0])
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calculated by autodiff for single y-vector")
        # Test that we can evaluate the kernel gradient with multiple y-samples
        k_ad = self.kernel.eval(x_ad, self.y)
        dk_ad = ad.ExtractGradient(k_ad)
        # Evaluate the gradient directly
        dk = self.kernel.gradient(self.x[:, 2], self.y)
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calcuated by autodiff for multiple y-vectors")

    def test_eval_autodiff(self):
        """ Test evaluating the kernel when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single sample y-point
        Kxy_ad = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1, :1], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is vector")
        # Test with multiple input sample y-points
        Kxy_ad = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1,:], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is array")

    def test_gradient_autodiff(self):
        """Test evaluating the kernel gradient when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single y-point
        dKxy_ad  =self.kernel.gradient(x_ad, self.y[:, 0])
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is a single vector")
        # Test with multiple y-points
        dKxy_ad = self.kernel.gradient(x_ad, self.y)
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is multiple vectors")

class PolynoimalKernelTest(unittest.TestCase):
    """Unittest for the PolynomialKernel Class"""
    def setUp(self):
        W = np.eye(2)
        offset = np.ones((1,))
        self.kernel = kernels.PolynomialKernel(weights=W, offset=offset, degree=2)
        self.x = np.array([[1, 0], [0, 1], [-1, 1]]).T
        self.y = np.array([[2, -1], [-3, 2]]).T
        self.Kxy_expected = np.array([[3., 0., -2.],[-2., 3., 6.]]).T**2
        self.Kxx_expected = np.array([[2., 1., 0.], [1., 2., 2.], [0., 2., 3.]]).T**2

    def test_eval(self):
        """Check the evaluation of the kernel function"""
        Kxy = self.kernel.eval(self.x, self.y)
        np.testing.assert_allclose(Kxy, self.Kxy_expected, atol=1e-12, err_msg=f"Asymmetric Kernel evaluation produces incorrect kernel matrix")
        Kxx = self.kernel.eval(self.x, self.x)
        np.testing.assert_allclose(Kxx, self.Kxx_expected, atol=1e-12, err_msg=f"Symmetric Kernel evaluation produces incorrect kernel matrix")

    def test_gradient(self):
        """Test evaluation of the kernel gradients"""
        x_ad = np.squeeze(ad.InitializeAutoDiff(self.x[:, 2]))
        # Test evaluating the kernel gradient with a single y-sample
        k_ad = self.kernel.eval(x_ad, self.y[:, 0])
        dk_ad = ad.ExtractGradient(k_ad)
        dk = self.kernel.gradient(self.x[:, 2], self.y[:, 0])
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calculated by autodiff for single y-vector")
        # Test that we can evaluate the kernel gradient with multiple y-samples
        k_ad = self.kernel.eval(x_ad, self.y)
        dk_ad = ad.ExtractGradient(k_ad)
        # Evaluate the gradient directly
        dk = self.kernel.gradient(self.x[:, 2], self.y)
        np.testing.assert_allclose(dk, dk_ad, atol=1e-6, err_msg=f"Gradient calculation does not match gradient calcuated by autodiff for multiple y-vectors")

    def test_eval_autodiff(self):
        """ Test evaluating the kernel when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single sample y-point
        Kxy_ad = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1, :1], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is vector")
        # Test with multiple input sample y-points
        Kxy_ad = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(ad.ExtractValue(Kxy_ad), self.Kxy_expected[:1,:], atol=1e-6, err_msg=f"Kernel evaluation fails when first argument is autodiff, second argument is array")

    def test_gradient_autodiff(self):
        """Test evaluating the kernel gradient when one of the inputs is an autodiff type"""
        x_ad = ad.InitializeAutoDiff(self.x[:, 0])
        # Test with a single y-point
        dKxy_ad  =self.kernel.gradient(x_ad, self.y[:, 0])
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y[:, 0])
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is a single vector")
        # Test with multiple y-points
        dKxy_ad = self.kernel.gradient(x_ad, self.y)
        dKxy = ad.ExtractValue(dKxy_ad)
        gradKxy = self.kernel.eval(x_ad, self.y)
        np.testing.assert_allclose(dKxy, ad.ExtractGradient(gradKxy), atol=1e-6, err_msg=f"Evaluating gradient fails when first input is autodiff, second input is multiple vectors")   

if __name__ == '__main__':
    unittest.main()