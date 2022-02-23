"""
unittests for pycito.systems.contactmodel.py

Luke Drnach
February 23, 2022
"""

import unittest
import numpy as np

import pydrake.autodiffutils as ad
import pycito.systems.contactmodel as cm

class OrthogonalizationTest(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 0, 0])
        self.y = np.array([0, 1, 0])
        self.z = np.array([0, 0, 1])
    
    def test_orthogonalization_xaxis(self):
        # Test with the x-axis as the normal
        t, b = cm.householderortho3D(self.x)
        np.testing.assert_allclose(t, self.y, atol=1e-6, err_msg="Orthogonalization returns incorrect tangent vector for x-axis normal")
        np.testing.assert_allclose(b, self.z, atol=1e-6, err_msg="Orthogonalization returns incorrect binormal vector for x-axis normal")

    def test_orthogonalization_yaxis(self):
        # Test with the y-axis as the normal
        t, b = cm.householderortho3D(self.y)
        np.testing.assert_allclose(t, -self.x, atol=1e-6, err_msg="Orthogonalization returns incorrect tangent vector for y-axis normal")
        np.testing.assert_allclose(b, self.z, atol=1e-6, err_msg="Orthogonalization returns incorrect binormal vector for y-axis normal")

    def test_orthogonalization_zaxis(self):
        # Test with the y-axis as the normal
        t, b = cm.householderortho3D(self.z)
        np.testing.assert_allclose(t, self.y, atol=1e-6, err_msg="Orthogonalization returns incorrect tangent vector for y-axis normal")
        np.testing.assert_allclose(b, -self.x, atol=1e-6, err_msg="Orthogonalization returns incorrect binormal vector for y-axis normal")

    def test_orthogonalization_arbitrary(self):
        # Test with an arbitrary unit normal vector
        test_normal = np.array([2, 3, -1])
        test_normal = test_normal / np.linalg.norm(test_normal)
        t, b = cm.householderortho3D(test_normal)
        R = np.column_stack([test_normal, t, b])
        np.testing.assert_allclose(R.dot(R.T), np.eye(3), atol=1e-12, err_msg=f"Orthgonalization does not return an orthogonal matrix for an arbitrary normal vector")
        d = np.linalg.det(R)
        self.assertAlmostEqual(d, 1., places = 12, msg="Orthogonalization returns rotation matrix with non-unit determinant")

class ConstantModelTest(unittest.TestCase):
    def setUp(self):
        self.c = 2
        self.fun = cm.ConstantModel(const = self.c)
        self.test_vals = [np.array([2]), 
                        np.array([1, 2]),
                        np.array([2, 4, -6]),
                        np.array([1, -2, 3])]

    def test_evaluation(self):
        # Test evaluation of the constant model
        expected = np.array([self.c])
        for test in self.test_vals:
            out = self.fun(test)
            np.testing.assert_allclose(out, expected, atol=1e-12, err_msg=f"Constant model evaluation failed on input {test}")

    def test_gradient(self):
        # Test evaluation of the gradient of the constant model - test against autodiff
        for test in self.test_vals:
            dout = self.fun.gradient(test)
            t_ad = np.squeeze(ad.InitializeAutoDiff(test))
            out_ad = self.fun(t_ad)
            dout_ad = out_ad.derivatives()
            np.testing.assert_allclose(dout, dout_ad, atol=1e-12, err_msg=f"Gradients for constant model do not match autodiff gradients on input {test}")

class FlatModelTest(unittest.TestCase):
    def setUp(self):
        self.loc = 3
        self.dir = np.array([0, 0, 1])
        self.fun = cm.FlatModel(location = self.loc, direction=self.dir)
        self.test_vals = [np.array([1, 2, 3]), 
                        np.array([5, -3, -1]),
                        np.array([-4, 2, 0]),
                        np.array([0, 0, 1]),
                        np.array([1, 5, 7])]
        self.test_ans = [np.array([val]) for val in [0, -4, -3, -2, 4]]

    def test_evaluation(self):
        """Test evaluation of the Flat Model"""
        for test, expected in zip(self.test_vals, self.test_ans):
            out = self.fun.eval(test)
            np.testing.assert_allclose(out, expected, atol=1e-12, err_msg=f"FlatModel fails to evaluate {test} example correctly")

    def test_gradient(self):
        for test in self.test_vals:
            grad = self.fun.gradient(test)
            test_ad = np.squeeze(ad.InitializeAutoDiff(test))
            eval_ad = self.fun.eval(test_ad)
            np.testing.assert_allclose(grad, eval_ad.derivatives(), atol=1e-12, err_msg=f"FlatModel gradient does not match autodiff gradient on {test} example")

class SemiparametricModelTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_prior_evaluation(self):
        pass

    def test_prior_gradient(self):
        pass

    def test_posterior_evaluation(self):
        pass

    def test_posterior_gradient(self):
        pass

    def test_add_samples(self):
        pass

    def test_model_errors(self):
        pass

class ContactModelTest(unittest.TestCase):
    pass

class SemiparametricContactModelTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()