"""
unittests for contactimplicit.py

Luke Drnach
October 14, 2020
"""

import numpy as np
import unittest
import pycito.systems.A1.a1 as a1_models
from pydrake.all import RigidTransform
from abc import ABC, abstractmethod

class _A1TestBase(ABC):
    def setUp(self):
        self.a1 = None
        self.make_a1_model()
        self.expected_num_positions = 19
        self.expected_num_velocities = 18

    @abstractmethod
    def make_a1_model(self):
        pass

    def test_num_positions(self):
        """Assert the model has the expected number of position variables"""
        self.assertEqual(self.a1.multibody.num_positions(), self.expected_num_positions, msg=f"Expected {self.expected_num_positions} position variables, got {self.a1.multibody.num_positions()} for {self.a1.__class__}")

    def test_num_velocities(self):
        """Assert the model has the expected number of velocity variables"""
        self.assertEqual(self.a1.multibody.num_velocities(), self.expected_num_velocities, msg=f"Expected {self.expected_num_velocities} velocity variables, got {self.a1.multibody.num_velocities()} for {self.a1.__class__}")

class A1FloatingBaseTest(_A1TestBase, unittest.TestCase):
    def setUp(self):
        super(A1FloatingBaseTest, self).setUp()

    def make_a1_model(self):
        self.a1 = a1_models.A1()
        self.a1.Finalize()

class A1VirtualBaseTest(_A1TestBase, unittest.TestCase):
    def setUp(self):
        super(A1VirtualBaseTest, self).setUp()
        self.expected_num_positions -= 1

    def make_a1_model(self):
        self.a1 = a1_models.A1VirtualBase()
        self.a1.Finalize()

class A1PlanarTest(_A1TestBase, unittest.TestCase):
    def setUp(self):
        super(A1PlanarTest, self).setUp()
        self.expected_num_positions = 15
        self.expected_num_velocities = 15

    def make_a1_model(self):
        self.a1 = a1_models.PlanarA1()
        self.a1.Finalize()

if __name__ == "__main__":
    unittest.main()