"""
Integration test for contact implicit methods using the sliding block

Luke Drnach
October 15, 2021
"""

import unittest
import numpy as np
import trajopt.contactimplicit as ci
from systems.block.block import Block

class BlockTestBase(unittest.TestCase):
    @staticmethod
    def get_boundary_points():
            return np.array([0, 0.5, 0., 0.]), np.array([5., 0.5, 0., 0.])

    def create_sliding_block(self):
        self.plant = Block()
        self.plant.Finalize()

    def create_block_trajopt(self):
        context = self.plant.multibody.CreateDefaultContext()
        options = ci.OptimizationOptions()
        self.trajopt = ci.ContactImplicitDirectTranscription(self.plant, context,
                                                        num_time_samples = 101,
                                                        maximum_timestep = 0.01,
                                                        minimum_timestep = 0.01,
                                                        options = options)
    
    def add_boundary_constraints(self,x0, xf):
        self.trajopt.add_state_constraint(knotpoint=0, value=x0)    
        self.trajopt.add_state_constraint(knotpoint=self.x.shape[1]-1, value=xf)

    def add_control_cost(self):
        nU = self.trajopt.u.shape[0]
        R = 10 * np.eye(nU)
        b = np.zeros((nU,))
        self.trajopt.add_quadratic_running_cost(R, b, [self.trajopt.u], name="ControlCost")
        return self.trajopt

    def add_state_cost(self, xf):
        nX = self.trajopt.x.shape[0]
        R = np.eye(nX)
        self.trajopt.add_quadratic_running_cost(R, xf, [self.trajopt.x], name="StateCost")

    def add_linear_guess(self, x0, xf):
        # Set the initial trajectory guess
        u_init = np.zeros(self.trajopt.u.shape)
        x_init = np.zeros(self.trajopt.x.shape)
        for n in range(0, x_init.shape[0]):
            x_init[n,:] = np.linspace(start=x0[n], stop=xf[n], num=101)
        l_init = np.zeros(self.trajopt.l.shape)
        # Set the guess in the trajopt
        self.trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
        return self.trajopt

    def set_snopt_options(self):
        options = {"Iterations limit": 10000,
                    "Major feasibility tolerance": 1e-6,
                    "Major optimality tolerance": 1e-6,
                    "Scale option": 2}
        self.trajopt.useSnoptSolver()
        self.trajopt.setSolverOptions(options)

    def setUp(self):
        self.create_sliding_block()
        x0, xf = self.get_boundary_points()
        self.create_block_trajopt()
        self.add_boundary_constraints(x0, xf)
        self.add_control_cost()
        self.add_state_cost(xf)
        self.add_linear_guess(x0, xf)
        self.set_snopt_options()
class SlidingBlockDirectTest(BlockTestBase):
    def setUp(self):
        super(SlidingBlockDirectTest, self).setUp()
        
    def test_trajopt(self):
        result = self.trajopt.solve()
        self.assertTrue(result.is_success(), msg = "Optimization failed")

class SlidingBlockCollocationTest(BlockTestBase):
    def create_block_trajopt(self):
        context = self.plant.multibody.CreateDefaultContext()
        options = ci.OrthogonalOptimizationOptions()
        options.useComplementarityWithCost()
        N = 31
        maxtime = 1
        mintime = 1
        self.trajopt = ci.ContactImplicitOrthogonalCollocation(self.plant, context,
                                                        num_time_samples = 31,
                                                        maximum_timestep = maxtime/(N-1),
                                                        minimum_timestep = mintime/(N-1),
                                                        state_order=3,
                                                        options = options)
    def add_control_cost(self):
        nU = self.trajopt.u.shape[0]
        R = 10 * np.eye(nU)
        b = np.zeros((nU,))
        self.trajopt.add_quadratic_control_cost(R, b, [self.trajopt.u], name="ControlCost")
        return self.trajopt
    
    def setUp(self):
        super(SlidingBlockCollocationTest, self).setUp()
        
    def test_trajopt(self):
        result = self.trajopt.solve()
        self.assertTrue(result.is_success(), msg = "Optimization failed")