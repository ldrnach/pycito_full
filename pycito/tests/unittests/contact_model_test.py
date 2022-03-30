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
        np.testing.assert_allclose(t, -self.z, atol=1e-6, err_msg="Orthogonalization returns incorrect tangent vector for x-axis normal")
        np.testing.assert_allclose(b, self.y, atol=1e-6, err_msg="Orthogonalization returns incorrect binormal vector for x-axis normal")

    def test_orthogonalization_yaxis(self):
        # Test with the y-axis as the normal
        t, b = cm.householderortho3D(self.y)
        np.testing.assert_allclose(t, -self.z, atol=1e-6, err_msg="Orthogonalization returns incorrect tangent vector for y-axis normal")
        np.testing.assert_allclose(b, -self.x, atol=1e-6, err_msg="Orthogonalization returns incorrect binormal vector for y-axis normal")

    def test_orthogonalization_zaxis(self):
        # Test with the y-axis as the normal
        t, b = cm.householderortho3D(self.z)
        np.testing.assert_allclose(t, self.x, atol=1e-6, err_msg="Orthogonalization returns incorrect tangent vector for y-axis normal")
        np.testing.assert_allclose(b, self.y, atol=1e-6, err_msg="Orthogonalization returns incorrect binormal vector for y-axis normal")

    def test_orthogonalization_arbitrary(self):
        # Test with an arbitrary unit normal vector
        test_normal = np.array([2, 3, -1])
        test_normal = test_normal / np.linalg.norm(test_normal)
        t, b = cm.householderortho3D(test_normal)
        R = np.column_stack([test_normal, t, b])
        np.testing.assert_allclose(R.dot(R.T), np.eye(3), atol=1e-12, err_msg=f"Orthgonalization does not return an orthogonal matrix for an arbitrary normal vector")
        d = np.linalg.det(R)
        self.assertAlmostEqual(d, 1., places = 12, msg="Orthogonalization returns rotation matrix with non-unit determinant")

    def test_orthogonalization_autodiff(self):
        """Test that we can evaluate orthogonalization using autodiff types"""
        x_ad = ad.InitializeAutoDiff(self.x)
        t, b = cm.householderortho3D(x_ad)
        # Check that the *values* are correct. We don't need to check the value of the gradients
        np.testing.assert_allclose(np.squeeze(ad.ExtractValue(t)), -self.z, atol=1e-6, err_msg='Orthogonalization returns inaccurate tangent vector when input is autodiff type')
        np.testing.assert_allclose(np.squeeze(ad.ExtractValue(b)), self.y, atol=1e-6, err_msg="Orthogonalization returns inaccurate binormal vector when input is autodiff type")

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
            dout_ad = ad.ExtractGradient(out_ad)
            np.testing.assert_allclose(dout, dout_ad, atol=1e-12, err_msg=f"Gradients for constant model do not match autodiff gradients on input {test}")

    def test_eval_autodiff(self):
        """Test that we can evaluate the function with autodiff types"""
        test = ad.InitializeAutoDiff(self.test_vals[2])
        expected = np.array([self.c])
        eval_ad = self.fun.eval(test)
        np.testing.assert_allclose(ad.ExtractValue(eval_ad).flatten(), expected, atol=1e-12, err_msg=f"ConstantModel fails to evaluate autodiff type correctly")

    def test_gradient_autodiff(self):
        """Test that we can evaluate the gradient using autodiff types"""
        test = ad.InitializeAutoDiff(self.test_vals[2])
        expected = np.zeros((3,))
        grad_ad = self.fun.gradient(test)
        np.testing.assert_allclose(np.squeeze(ad.ExtractValue(grad_ad)), expected, atol=1e-12, err_msg=f"ConstantModel fails to correctly evaluate gradient with autodiff type")

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
            test_ad = ad.InitializeAutoDiff(test)
            eval_ad = self.fun.eval(test_ad)
            np.testing.assert_allclose(grad, ad.ExtractGradient(eval_ad), atol=1e-12, err_msg=f"FlatModel gradient does not match autodiff gradient on {test} example")

    def test_eval_autodiff(self):
        """Test that we can evaluate the function with autodiff types"""
        test = ad.InitializeAutoDiff(self.test_vals[0])
        expected = self.test_ans[0]
        eval_ad = self.fun.eval(test)
        np.testing.assert_allclose(ad.ExtractValue(eval_ad).flatten(), expected, atol=1e-12, err_msg=f"FlatModel fails to evaluate autodiff type correctly")

    def test_gradient_autodiff(self):
        """Test that we can evaluate the gradient using autodiff types"""
        test = ad.InitializeAutoDiff(self.test_vals[0])
        expected = np.reshape(self.dir, (1,3))
        grad_ad = self.fun.gradient(test)
        np.testing.assert_allclose(ad.ExtractValue(grad_ad), expected, atol=1e-12, err_msg=f"FlatModel fails to correctly evaluate gradient with autodiff type")

class ContactModelTest(unittest.TestCase):
    def setUp(self):
        self.model = cm.ContactModel.FlatSurfaceWithConstantFriction(friction = 0.5)
        self.test_points = [np.array([1, 2, 3]), 
                            np.array([-1, 0, -3]),
                            np.array([0, 1, 0]),
                            np.array([-4, 3, 2])]
        self.expected_surfs = [np.array([val]) for val in [3, -3, 0, 2]]
        self.expected_friction = np.array([0.5])
        self.expected_frame = np.array([[0, 0, 1],
                                        [1, 0, 0],
                                        [0, 1, 0]])
    def test_surface_evaluation(self):
        """Test contact model surface evaluation"""
        for test, expected in zip(self.test_points, self.expected_surfs):
            surf = self.model.eval_surface(test)
            np.testing.assert_allclose(surf, expected, atol=1e-12, err_msg=f"Evaluating contact model surface inaccurate for test case {test}")

    def test_friction_evaluation(self):
        """Test contact model friction evaluation"""
        for test in self.test_points:
            fric = self.model.eval_friction(test)
            np.testing.assert_allclose(fric, self.expected_friction, atol=1e-12, err_msg=f"Evaluating contact model friction inaccurate for test point {test}")

    def test_local_frame(self):
        """Test contact model surface local frame evaluation"""
        for test in self.test_points:
            frame = self.model.local_frame(test)
            np.testing.assert_allclose(frame, self.expected_frame, atol=1e-12, err_msg=f"Evaluating local frame inaccurate at test point {test}")

    def test_surface_eval_autodiff(self):
        """Test evaluating the contact surface with autodiff types"""
        test = ad.InitializeAutoDiff(self.test_points[0])
        expected = self.expected_surfs[0]
        surf_ad = self.model.eval_surface(test)
        np.testing.assert_allclose(ad.ExtractValue(surf_ad).flatten(), expected, atol=1e-12, err_msg=f"Contact Model surface evaluation inaccurate using autodiff types")

    def test_friction_eval_autodiff(self):
        """Test evaluating the friction coefficient with autodiff types"""
        test = ad.InitializeAutoDiff(self.test_points[0])
        expected = self.expected_friction
        fric_ad = self.model.eval_friction(test)
        np.testing.assert_allclose(ad.ExtractValue(fric_ad).flatten(), expected, atol=1e-12, err_msg=f"Contact model evaluating friction using autodiffs is inaccurate")

    def test_local_frame_autodiff(self):
        """Test evaluating the local frame using autodiffs"""
        test = ad.InitializeAutoDiff(self.test_points[0])
        frame_ad = self.model.local_frame(test)
        np.testing.assert_allclose(ad.ExtractValue(frame_ad), self.expected_frame, atol=1e-12, err_msg=f"Contact model local frame evaluation inaccurate when using autodiff types")

class SemiparametricModelTest(unittest.TestCase):
    def setUp(self):
        self.model = cm.SemiparametricModel.ConstantPriorWithRBFKernel(const = 0, length_scale=1)
        self.data = np.array([[1, 0, 0],
                             [2, 0, -1]]).T 
        self.weights = np.array([0, 1]) 
        # Test case data
        self.test_point = np.array([1, 1, 0])  
        self.expected_posterior = np.array([np.exp(-3/2)])
        self.expected_errors = np.array([np.exp(-1), 1])

    def test_prior_evaluation(self):
        """Test evaluating the model before data has been added"""
        eval = self.model.eval(self.test_point)
        np.testing.assert_allclose(eval, np.zeros((1,)), atol=1e-12, err_msg=f"Semiparametric model fails to evaluate the prior accurately")

    def test_prior_gradient(self):
        """Test that we can evaluate the model gradient before data has been added"""
        grad = self.model.gradient(self.test_point)
        np.testing.assert_allclose(grad, np.zeros((1,3)), atol=1e-12, err_msg=f"Semiparametric model fails to evaluate the prior gradient accurately")

    def test_posterior_evaluation(self):
        """Test evaluating the model after data has been added"""
        self.model.add_samples(self.data, self.weights)
        eval = self.model.eval(self.test_point)
        np.testing.assert_allclose(eval, self.expected_posterior, atol=1e-8, err_msg=f"Semiparametric model fails to evaluate the posterior accurately")
        
    def test_posterior_gradient(self):
        """
        Test evaluating the model gradient after data has been added
        Check gradient calculation against autodiff type
        """
        self.model.add_samples(self.data, self.weights)
        grad = self.model.gradient(self.test_point)
        test_ad = ad.InitializeAutoDiff(self.test_point)
        grad_expected = ad.ExtractGradient(self.model.eval(test_ad))
        np.testing.assert_allclose(grad, grad_expected, atol=1e-8, err_msg=f"Semiparametric model fails to evaluate the posterior gradient accurately")

    def test_add_samples(self):
        """Check that adding sample points to the model works"""
        self.assertEqual(self.model.num_samples, 0, msg=f"Nonzero number of example points before adding data to the model")
        self.model.add_samples(self.data, self.weights)
        self.assertEqual(self.model.num_samples, 2, msg=f"Incorrect number of example points added")

    def test_model_errors(self):
        """Check that we can get the modeling errors accurately"""
        self.model.add_samples(self.data, self.weights)
        err = self.model.model_errors
        np.testing.assert_allclose(err, self.expected_errors, atol=1e-8, err_msg=f"Calculating model errors is inaccurate")

    def test_posterior_autodiff(self):
        """
        Check that we can evaluate the posterior using autodiffs
        Check against the gradient calculated by evaluating *eval* with autodiff types
        """
        self.model.add_samples(self.data, self.weights)
        test_ad = ad.InitializeAutoDiff(self.test_point)
        eval_ad = self.model.eval(test_ad)
        self.assertEqual(eval_ad.shape, self.expected_posterior.shape, msg="Evaluating posterior with autodiff returns the wrong shape")
        # NOTE: ad.Extract value returns a (n,1) vector even when the input is (n,). Thus, flatten is needed to correct for this behavior
        np.testing.assert_allclose(ad.ExtractValue(eval_ad).flatten(), self.expected_posterior, atol=1e-8, err_msg=f"Semiparametric model fails to evaluate posterior correctly using autodiff types")

    def test_posterior_gradient_autodiff(self):
        """
        Check that we can evaluate the posterior gradient using autodiffs
        Check against the gradient calculated using *eval* with autodiff types
        """
        self.model.add_samples(self.data, self.weights)
        test_ad = ad.InitializeAutoDiff(self.test_point)
        grad_ad = self.model.gradient(test_ad)
        eval_ad = self.model.eval(test_ad)
        np.testing.assert_allclose(ad.ExtractValue(grad_ad), ad.ExtractGradient(eval_ad), atol=1e-8, err_msg=f"Semiparametric model fails to evaluate posterior correctly using autodiff types")


class SemiparametricContactModelTest(unittest.TestCase):
    def setUp(self):
        self.model = cm.SemiparametricContactModel.FlatSurfaceWithRBFKernel(height = 1.0, friction=1.0, length_scale=1.)
        self.data = np.array([[1, 0, 0],
                        [2, 0, -1]]).T 
        self.weights = np.array([0, 1]) 
        # Test case data
        self.test_point = np.array([1, 1, 0])  
        self.expected_surface_posterior = np.array([np.exp(-3/2) - 1])
        self.expected_friction_posterior = np.array([1 + np.exp(-3/2)])

    def test_surface_eval(self):
        """Test that we can evaluate the surface function accurately before and after adding points to the model"""
        # Test evaluating the surface prior
        eval = self.model.eval_surface(self.test_point)
        np.testing.assert_allclose(eval, -np.ones((1,)), atol=1e-12, err_msg=f"Semiparametric model fails to evaluate the surface prior accurately for float types")
        # Test evaluating the surface posterior
        self.model.add_samples(self.data, self.weights, self.weights)
        eval = self.model.eval_surface(self.test_point)
        np.testing.assert_allclose(eval, self.expected_surface_posterior, atol=1e-8, err_msg=f"Semiparametric model fails to evaluate the surface posterior accurately for float types")

    def test_friction_eval(self):
        """Test that we can evaluate the friction function accurately before and after adding points to the model"""
        # Test evaluating the friction prior
        eval = self.model.eval_friction(self.test_point)
        np.testing.assert_allclose(eval, np.ones((1,)), atol=1e-12, err_msg=f"Semiparametric model fails to evaluate the friction prior accurately for float types")
        # Test evaluating the friction posterior
        self.model.add_samples(self.data, self.weights, self.weights)
        eval = self.model.eval_friction(self.test_point)
        np.testing.assert_allclose(eval, self.expected_friction_posterior, atol=1e-8, err_msg=f"Semiparametric model fails to evaluate the friction posterior accurately for float types")

    def test_local_frame(self):
        """Test that we can evaluate the local frame before and after adding points to the model"""
        # Test evaluating the local frame before adding data
        R = self.model.local_frame(self.test_point)
        R_expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(R, R_expected, atol=1e-6, err_msg=f"Evaluating prior local frame fails for float types")
        # Test evaluating the local frame after adding data
        self.model.add_samples(self.data, self.weights, self.weights)
        R = self.model.local_frame(self.test_point)
        np.testing.assert_allclose(R.dot(R.T), np.eye(3), atol=1e-6, err_msg=f"Evaluating posterior local frame does not return an orthonormal matrix for float types")
        
    def test_surface_eval_autodiff(self):
        """Test that we can evaluate the surface using autodiff types"""
        # Test evaluating the surface prior
        test_ad = ad.InitializeAutoDiff(self.test_point)
        eval = self.model.eval_surface(test_ad)
        np.testing.assert_allclose(ad.ExtractValue(eval).flatten(), -np.ones((1,)), atol=1e-12, err_msg=f"Semiparametric model fails to evaluate the surface prior accurately using autodiff types")
        # Test evaluating the surface posterior
        self.model.add_samples(self.data, self.weights, self.weights)
        eval = self.model.eval_surface(test_ad)
        np.testing.assert_allclose(ad.ExtractValue(eval).flatten(), self.expected_surface_posterior, atol=1e-8, err_msg=f"Semiparametric model fails to evaluate the surface posterior accurately using autodiff types")

    def test_friction_eval_autodiff(self):
        """Test that we can evaluate the friction function accurately before and after adding points to the model"""
        # Test evaluating the friction prior
        test_ad = ad.InitializeAutoDiff(self.test_point)
        eval = self.model.eval_friction(test_ad)
        np.testing.assert_allclose(ad.ExtractValue(eval).flatten(), np.ones((1,)), atol=1e-12, err_msg=f"Semiparametric model fails to evaluate the friction prior accurately using autodiff types")
        # Test evaluating the friction posterior
        self.model.add_samples(self.data, self.weights, self.weights)
        eval = self.model.eval_friction(test_ad)
        np.testing.assert_allclose(ad.ExtractValue(eval).flatten(), self.expected_friction_posterior, atol=1e-8, err_msg=f"Semiparametric model fails to evaluate the friction posterior accurately using autodiff types")

    def test_local_frame_autodiff(self):
        """Test that we can evaluate the local frame before and after adding points to the model"""
        # Test evaluating the local frame before adding data
        test_ad = ad.InitializeAutoDiff(self.test_point)
        R = self.model.local_frame(test_ad)
        R_expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(ad.ExtractValue(R), R_expected, atol=1e-6, err_msg=f"Evaluating prior local frame fails for autodiff types")
        # Test evaluating the local frame after adding data
        self.model.add_samples(self.data, self.weights, self.weights)
        R_ad = self.model.local_frame(test_ad)
        R = ad.ExtractValue(R_ad)
        np.testing.assert_allclose(R.dot(R.T), np.eye(3), atol=1e-6, err_msg=f"Evaluating posterior local frame does not return an orthonormal matrix for autodiff types")
        

if __name__ == '__main__':
    unittest.main()