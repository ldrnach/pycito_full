"""
Unittests for contact constraints in trajopt.constraints

Luke Drnach
January 31, 2022
"""
#TODO: Write unittest for MultibodyDynamicsConstraint
import unittest
import numpy as np

from pydrake.all import MathematicalProgram
import pydrake.autodiffutils as ad

from pycito.trajopt import constraints as cstr
from pycito.systems.fallingrod.fallingrod import FallingRod

class NormalDistanceTest(unittest.TestCase):
    def setUp(self):
        # Create a plant model
        self.plant = FallingRod()
        self.plant.Finalize()
        # Create the constraint
        self.cstr_fcn = cstr.NormalDistanceConstraint(self.plant)
        # Create a mathematical program
        self.prog = MathematicalProgram()
        # Create the example states
        self.x0 = np.zeros((6,))
        self.x1  = np.array([1, 2, np.pi/6, 0.5, 0, 0.1])
        # Check the expected distance
        self.expected_dist = np.array([2 - 0.25 * np.cos(np.pi/6) - 0.05, 2 + 0.25 * np.cos(np.pi/6) - 0.05])
        self.expected_dist_zero = np.array([-0.25-0.05, 0.25-0.05])
        # The linearization
        self.A_expected = np.array([[0, 1, 0.25 * np.sin(np.pi/6), 0, 0, 0], 
                                   [0, 1, -0.25 * np.sin(np.pi/6), 0, 0, 0]])
                            
    def test_eval_float(self):
        """Check that we can evaluate the normal distance constraint"""
        # Test at the zero configuration
        dist0 = self.cstr_fcn(self.x0)
        np.testing.assert_allclose(dist0, self.expected_dist_zero, atol=1e-7, err_msg=f"Normal distances inaccurate at zero configuration")
        # Test at the nonzero configuration
        dist1 = self.cstr_fcn(self.x1)
        np.testing.assert_allclose(dist1, self.expected_dist, atol=1e-7, err_msg=f"Normal distance inaccurate at nonzero configuration")

    def test_add_to_program(self):
        """Check that we can add the constraint to the program"""
        # Create decision variables
        xvar = self.prog.NewContinuousVariables(rows=6, cols=1, name='state')
        # Check the number of constraints before
        numCstr_pre = len(self.prog.GetAllConstraints())
        self.assertEqual(numCstr_pre, 0, msg=f"Mathematical program has constraints before constraints are added")
        # Check that we can add the values after
        self.cstr_fcn.addToProgram(self.prog, xvar)
        binding = self.prog.GetAllConstraints()
        self.assertEqual(len(binding), 1, msg="NormalDistanceConstraint.addToProgram adds an unexpected number of constraints")
        # Check that we can evaluate the constraint
        dvals = np.zeros((6,))
        out = self.prog.EvalBinding(binding[0], dvals)
        np.testing.assert_allclose(out, self.expected_dist_zero, atol=1e-7, err_msg=f'Failed to correctly evaluate the NormalDistanceConstraint within MathematicalProgram')

    def test_eval_autodiff(self):
        """Check that we can evaluate the constraint using autodiff types"""
        x_ad = ad.InitializeAutoDiff(self.x1)
        dist_ad = self.cstr_fcn(x_ad)
        dist_val = np.squeeze(ad.autoDiffToValueMatrix(dist_ad))
        np.testing.assert_allclose(dist_val, self.expected_dist, atol=1e-7, err_msg=f"Evaluating NormalDistance with autodiff types produces incorrect results")

    def test_linearization(self):
        """Test the linearization of the constraint"""
        A, b = self.cstr_fcn.linearize(self.x1)
        np.testing.assert_allclose(b, self.expected_dist, atol=1e-7, err_msg=f"Linearization fails to accurately evaluate the constraint")
        np.testing.assert_allclose(A, self.A_expected, atol=1e-7, err_msg=f"Linearization fails to accurately evaluate the gradient of the constraint")

class DissipationTest(unittest.TestCase):
    def setUp(self):
        # Create a plant model
        self.plant = FallingRod()
        self.plant.Finalize()
        # Create the constraint
        self.cstr_fcn = cstr.MaximumDissipationConstraint(self.plant)
        # Create a mathematical program
        self.prog = MathematicalProgram()
        # Create the example states
        self.x0 = np.zeros((6,))
        self.x1  = np.array([1, 2, np.pi/6, 0.5, 0, 0.1])
        self.s0 = np.zeros((2,))
        self.s1 = np.array([1, 2])
        # Create the answers
        Jt1 = np.array([[1, 0, -0.25*np.cos(np.pi/6)],[0, 0, 0]], dtype=float)
        Jt2 = np.array([[1, 0, 0.25*np.cos(np.pi/6)],[0, 0, 0]], dtype=float)
        Jt = np.concatenate([Jt1, -Jt1, Jt2, -Jt2], axis=0)
        v = Jt.dot(self.x1[3:])
        s_expected = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        self.expected_diss = s_expected + v
        # Linearization
        self.A_expected = np.zeros((8, 8))
        self.A_expected[[0, 6], 2] = 0.25*np.sin(np.pi/6)*0.1
        self.A_expected[[2, 4], 2] = -0.25*np.sin(np.pi/6)*0.1
        self.A_expected[[0, 4], 3] = 1
        self.A_expected[[2, 6], 3] = -1
        self.A_expected[[0, 6], 5] = -0.25*np.cos(np.pi/6)
        self.A_expected[[2, 4], 5] = 0.25*np.cos(np.pi/6)
        self.A_expected[[0, 1, 2, 3], 6] = 1
        self.A_expected[[4, 5, 6, 7], 7] = 1

    def test_add_to_program(self):
        """Check that we can add the constraint to the program"""
        # Create desicion variables
        xvar = self.prog.NewContinuousVariables(rows=6, cols=1, name='state')
        svar = self.prog.NewContinuousVariables(rows=2, cols=1, name='sliding_slack')
        # Check the number of constraints before
        numCstr_pre = len(self.prog.GetAllConstraints())
        self.assertEqual(numCstr_pre, 0, msg=f"MathematicalProgram has constraints before constraints were added")
        # Check adding the constraint
        self.cstr_fcn.addToProgram(self.prog, xvar, svar)
        binding = self.prog.GetAllConstraints()
        self.assertEqual(len(binding), 1, msg="MaximumDissipationConstraint.addToProgram adds an unexpected number of constraints")
        # Check evaluating the binding
        dvals = np.zeros((8, ))
        out = self.prog.EvalBinding(binding[0], dvals)
        np.testing.assert_allclose(out, np.zeros((8,)), atol=1e-8, err_msg=f"Failed to evaluate maximum dissipation constraint within MathematicalProgram at zero configuration")

    def test_eval_float(self):
        """Check that we can evaluate the maximum dissipation constraint"""
        # Test at the zero configuration
        diss0 = self.cstr_fcn(np.concatenate([self.x0, self.s0], axis=0))
        np.testing.assert_allclose(diss0, np.zeros((8,)), atol=1e-7, err_msg=f"Maximum Dissipation inaccurate at zero configuration")
        # Test at a nonzero configuration
        diss = self.cstr_fcn(np.concatenate([self.x1, self.s1], axis=0))
        np.testing.assert_allclose(diss, self.expected_diss, atol=1e-7, err_msg=f"Maximum Dissipation inaccurate at nonzero configuration")

    def test_eval_autodiff(self):
        """Check that we can evaluate"""
        dvals = np.concatenate([self.x1, self.s1], axis=0)
        advals = ad.InitializeAutoDiff(dvals)
        diss_ad = self.cstr_fcn(advals)
        diss_vals = np.squeeze(ad.autoDiffToValueMatrix(diss_ad))
        np.testing.assert_allclose(diss_vals, self.expected_diss, atol=1e-7, err_msg=f"Evaluating MaximumDissipationConstraint with autodiff type produces inaccurate resutls")

    def test_linearization(self):
        """Test that the linearization is accurate"""
        A, b = self.cstr_fcn.linearize(self.x1, self.s1)
        np.testing.assert_allclose(b, self.expected_diss, atol=1e-7, err_msg=f"Linearization fails to evaluate the constraint accurately")
        np.testing.assert_allclose(A, self.A_expected, atol=1e-7, err_msg=f"Linearization fails to evaluate the gradient accurately")

class FrictionConeTest(unittest.TestCase):
    def setUp(self):
        # Create a plant model
        self.plant = FallingRod()
        self.plant.terrain.friction = 0.5
        self.plant.Finalize()
        # Create the constraint
        self.cstr_fcn = cstr.FrictionConeConstraint(self.plant)
        # Create a mathematical program
        self.prog = MathematicalProgram()
        # Create the example states
        self.x0 = np.zeros((6,))
        self.x1 = np.array([1, 2, np.pi/6, 0.5, 0, 0.1])
        self.fn = np.array([10, 5])
        self.ft = np.array([1, 0, 2, 1, 2, -1, 0, 3])
        self.friccone_expected = np.array([1, -1.5])
        self.A_expected = np.array([[0, 0, 0, 0, 0, 0, 0.5, 0.0, -1, -1, -1, -1, 0., 0., 0., 0.],
                                    [0, 0, 0, 0, 0, 0, 0.0, 0.5, 0., 0., 0., 0., -1, -1, -1, -1]])

    def test_add_to_program(self):
        """Check that we can add the constraint to the program"""
        # Create decision variables
        xvar = self.prog.NewContinuousVariables(rows=6, cols=1, name='state')
        fnvar = self.prog.NewContinuousVariables(rows=2, cols=1, name='normal_force')
        ftvar = self.prog.NewContinuousVariables(rows=8, cols=1, name='friction_force')
        # Check the number of constraints before
        numcstr_pre = len(self.prog.GetAllConstraints())
        self.assertEqual(numcstr_pre, 0, msg=f'MathematicalProgram has constraints before constraints were added')
        # Check adding the constraint
        self.cstr_fcn.addToProgram(self.prog, xvar, fnvar, ftvar)
        binding = self.prog.GetAllConstraints()
        self.assertEqual(len(binding), 1, msg='FrictionConeConstraint.addToProgram adds an unexpected number of constraints')
        # Check evaluating the binding
        dvals = np.zeros((16, ))
        out = self.prog.EvalBinding(binding[0], dvals)
        np.testing.assert_allclose(out, np.zeros((2,)), atol=1e-7, err_msg=f"Evaluating FrictionConeConstraint within MathematicalProgram fails at zero guess")

    def test_eval_float(self):
        """Check that we can evaluate the friction cone constraint with floats"""
        dvals = np.concatenate([self.x1, self.fn, self.ft], axis=0)
        friccone = self.cstr_fcn(dvals)
        np.testing.assert_allclose(friccone, self.friccone_expected, atol=1e-7, err_msg=f"FrictionConeConstraint does not evaluate correctly")

    def test_eval_autodiff(self):
        """Check that we can evaluate the friction cone constraint with autodiffs"""
        dvals = np.concatenate([self.x1, self.fn, self.ft], axis=0)
        dvals_ad = ad.InitializeAutoDiff(dvals)
        friccone_ad = self.cstr_fcn(dvals_ad)
        # Get the values
        friccone_vals = np.squeeze(ad.autoDiffToValueMatrix(friccone_ad))
        # Check the values
        np.testing.assert_allclose(friccone_vals, self.friccone_expected, atol=1e-7, err_msg=f"Evalutaing FricconeConstraint with autodiff type produces inaccurate results")

    def test_linearization(self):
        """Test the linearization of the constraint"""
        A, b = self.cstr_fcn.linearize(self.x1, self.fn, self.ft)
        np.testing.assert_allclose(b, self.friccone_expected, atol=1e-7, err_msg=f"Linearization fails to evaluate the constraint accurately")
        np.testing.assert_allclose(A, self.A_expected, atol=1e-7, err_msg=f"Linearization failed to evaluate gradient accurately")

class NormalDissipationTest(unittest.TestCase):
    def setUp(self):
        # Create a plant model
        self.plant = FallingRod()
        self.plant.Finalize()
        # Create the constraint
        self.cstr_fcn = cstr.NormalDissipationConstraint(self.plant)
        # Create a mathematical program
        self.prog = MathematicalProgram()
        # Create the example states
        self.x0 = np.zeros((6,))
        self.x1 = np.array([1, 2, np.pi/6, 0.5, 0, 0.1])
        self.fn = np.array([10, 5])
        self.dissipation_expected = np.array([1/8, -1/16])
        self.A_expected = np.array([[0.,  0., np.sqrt(3)/8, 0, 10, 10/8,  1/80, 0.],
                                    [0.,  0., -np.sqrt(3)/16, 0, 5,-5/8,  0., -.2/16]])

    def test_add_to_program(self):
        """Check that we can add the constraint to the program"""
        # Create decision variables
        xvar = self.prog.NewContinuousVariables(rows=6, cols=1, name='state')
        fnvar = self.prog.NewContinuousVariables(rows=2, cols=1, name='normal_force')
        # Check the number of constraints before
        numcstr_pre = len(self.prog.GetAllConstraints())
        self.assertEqual(numcstr_pre, 0, msg=f'MathematicalProgram has constraints before constraints were added')
        # Check adding the constraint
        self.cstr_fcn.addToProgram(self.prog, xvar, fnvar)
        binding = self.prog.GetAllConstraints()
        self.assertEqual(len(binding), 1, msg='NormalDissipationConstraint.addToProgram adds an unexpected number of constraints')
        # Check evaluating the binding
        dvals = np.zeros((8, ))
        out = self.prog.EvalBinding(binding[0], dvals)
        np.testing.assert_allclose(out, np.zeros((2,)), atol=1e-7, err_msg=f"Evaluating NormalDissipationConstraint within MathematicalProgram fails at zero guess")

    def test_eval_float(self):
        """Check that we can evaluate the normal dissipation constraint with floats"""
        dvals = np.concatenate([self.x1, self.fn], axis=0)
        dissipation = self.cstr_fcn(dvals)
        np.testing.assert_allclose(dissipation, self.dissipation_expected, atol=1e-7, err_msg=f"NormalDissipationConstraint does not evaluate correctly")

    def test_eval_autodiff(self):
        """Check that we can evaluate the normal dissipation constraint with autodiffs"""
        dvals = np.concatenate([self.x1, self.fn,], axis=0)
        dvals_ad = ad.InitializeAutoDiff(dvals)
        dissipation_ad = self.cstr_fcn(dvals_ad)
        # Get the values
        dissipation_ad = np.squeeze(ad.autoDiffToValueMatrix(dissipation_ad))
        # Check the values
        np.testing.assert_allclose(dissipation_ad, self.dissipation_expected, atol=1e-7, err_msg=f"Evaluating NormalDissipation with autodiff type produces inaccurate results")


    def test_linearization(self):
        """Test the linearization of the constraint"""
        A, b = self.cstr_fcn.linearize(self.x1, self.fn)
        np.testing.assert_allclose(b, self.dissipation_expected, atol=1e-7, err_msg=f"Linearization fails to evaluate the constraint accurately")
        np.testing.assert_allclose(A, self.A_expected, atol=1e-7, err_msg=f"Linearization failed to evaluate gradient accurately")


if __name__ == "__main__":
    unittest.main()
