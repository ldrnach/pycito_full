"""
contactmodel.py: module for specifying arbitrary contact models for use with TimeSteppingRigidBodyPlant

Luke Drnach
February 16, 2022
"""

# TODO: rewrite models to always return numpy arrays
# TODO: Integrate with Timestepping
from __future__ import annotations

import abc
import copy
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import MathematicalProgram, Solve

import pycito.decorators as deco
import pycito.systems.kernels as kernels
from configuration.parametricmodel import (
    ConstantModelConfig,
    FlatModelConfig,
    PiecewiseModelConfig,
)

if __name__ == "__main__":
    print("Hello from contactmodel.py!")
