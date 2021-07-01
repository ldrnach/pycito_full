"""
General methods for creating optimizers for multibody systems

Luke Drnach
June 28, 2021
"""
#TODO: Make a configuration class to hold standard details, like cost weights, boundary constraints, SnoptOptions, etc. Make the class saveable and loadable
#TODO: Create special instances for A1 and sliding block, with routines for static, lifting, and walking for A1
#TODO: Write scripts to create configurations
#TODO: Write scripts to load a sequence of configurations, run the optimization, and save the output to a subdirectory (optimization problem/date/run number)
#TODO: To the output directory, save the configuration, the trajectory data, the figures, and the trajectory optimization report
#TODO: Calculate the number of major iterations required for optimization and save it to a text file

import numpy as np
import matplotlib.pyplot as plt
import abc, os
import pickle as pkl
# Custom imports
from trajopt import contactimplicit as ci
import utilities as utils
import decorators as deco

class OptimizationConfiguration():
    def __init__(self):
        """Set the default configuration variables for trajectory optimization"""
        # Trajectory optimization settings
        self.num_time_samples = None
        self.maximum_time = None
        self.minimum_time = None
        # Complementarity settings
        self.complementarity_cost_weight = None
        self.complementarity_slack = None
        # Boundary Conditions
        self.state_constraints = None
        # Control and State Cost Weights
        self.control_cost = None
        self.state_cost = None
        # Final cost weights
        self.final_time_cost = None
        self.final_state_cost = None
        # Initial guess type
        self.initial_guess_type = 'zeros'
        # Solver options
        self.solver_options = {}
    
    @classmethod
    def load(cls, filename=None):
        """Load a configuration file from disk"""
        # Check that the file exists
        filename = utils.FindResource(filename)
        # Load the configuration data from the file
        with(filename, 'rb') as input:
            config = pkl.load(input)
        # Return the new configuration
        return config

    def save(self, filename=None):
        """Save the current configuration file to the disk"""
        # Check that filename is not empty
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # Save the configuration to the file
        with(filename, 'wb') as output:
            pkl.dump(self, output, pkl.HIGHEST_PROTOCOL)

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
        pass