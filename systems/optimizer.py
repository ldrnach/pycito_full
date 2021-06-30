"""
General methods for creating optimizers for multibody systems

Luke Drnach
June 28, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import abc
# Custom imports
from trajopt import contactimplicit as ci
import utilities as utils
import decorators as deco

class SystemOptimizer(abc.ABC):
    def __init__(self):
        self.make_system()
        self.options = self.defaultOptimizationOptions()

    @abc.abstractmethod
    def make_system(self):
        raise NotImplementedError

    @staticmethod
    def defaultOptimizationOptions():
        return ci.OptimizationOptions()    
    
    def solve(self):
        result = self._solve_program()
        utils.printProgramReport(result, self.prog)
        return result

    @deco.timer
    def _solve_program(self):
        return self.solver(self.prog)

    def set_solver_options(self, **kwargs):
        pass

    def generateReport(self):
        