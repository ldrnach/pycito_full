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
        pass
    
    @abc.abstractmethod
    def make_system(self):
        raise NotImplementedError

    def solve(self):
        result = self._solve_program()
        utils.printProgramReport(result, self.prog)

    @deco.timer
    def _solve_program(self):
        return self.solver(self.prog)