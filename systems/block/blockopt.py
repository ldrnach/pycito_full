"""
Class Methods for running Sliding Block Trajectory Optimization

Luke Drnach
April 13, 2021
"""

# System Imports
import timeit
import numpy as np
import matplotlib.pyplot as plt
# Custom Imports
from trajopt.contactimplicit import ContactConstraintViewer, ContactImplicitDirectTranscription, OptimizationOptions
import utilities as utils
from systems.block.block import Block
# Pydrake Imports
from pydrake.solvers.snopt import SnoptSolver

class BlockOptimizer():
    def __init__(self, options=None, min_timestep=0.01, max_timestep=0.01, num_timepoints=101):
        # Create a block model
        plant = Block()
        plant.Finalize()
        context = plant.multibody.CreateDefaultContext()
        # Create a default options
        if options is None:
            options = BlockOptimizer.defaultOptimizationOptions()
        self.trajopt = ContactImplicitDirectTranscription(plant, context, num_timepoints, min_timestep, max_timestep, options)
        # Create default snopt options
        self.snoptions = {
            "Iterations Limit": 10000,
            "Major Feasibility Tolerance": 1e-6,
            "Major Optimality Tolerance": 1e-6,
            "Scale Option": 2
        }
        # Default initial and final conditions
        self._initial_condition = np.array([0., 0.5, 0., 0.])
        self._final_condition = np.array([5., 0.5, 0., 0.])
        # Default control and state cost weights
        self._control_cost = None
        self._state_cost = None

    @staticmethod
    def defaultOptimizationOptions():
        options = OptimizationOptions()
        options.useNonlinearComplementarityWithConstantSlack()
        return options
    
    def setBoundaryConditions(self, initial=[0., 0.5, 0., 0.], final=[5., 0.5, 0., 0.]):
        """ set the values for the boundary conditions """
        if type(initial) is list and len(initial) == 4:
            self._initial_condition = np.array(initial)
        elif type(initial) is np.ndarray and initial.shape[0] == 4:
            self._initial_condition = initial
        else:
            raise ValueError("initial must be a 4 element array or list")
        if type(final) is list and len(final) == 4:
            self._final_condition = np.array(final)
        elif type(final) is np.ndarray and final.shape[0] == 4:
            self._final_condition = final
        else:
            raise ValueError("final must be a 4 element array or list")

    def setControlWeights(self, weights=None, ref=None):
        # Check the input weights
        if weights is None:
            return
        numU = self.trajopt.u.shape[0]
        R, uref = self._check_quadratic_cost_weights(weights, ref, numU)
        # Store the values for the cost
        self._control_cost = (R, uref)

    def setStateWeights(self, weights=None, ref=None):
        if weights is None:
            return
        numX = self.trajopt.x.shape[0]
        Q, xref = self._check_quadratic_cost_weights(weights, ref, numX)
        # Store values for the cost
        self._state_cost = (Q, xref)

    def _check_quadratic_cost_weights(self, weights, ref, numvals):
        if type(weights) is list and len(weights) == numvals:
            W = np.diag(weights)
        elif type(weights) is np.ndarray and weights.shape[0] == numvals:
            if weights.ndim() == 1:
                W = np.diag(weights)
            elif weights.shape[1] == numvals:
                W = weights
            else:
                raise ValueError(f"weights must be either a square array or an 1D array or list with {numvals} elements")
        else:
            raise ValueError(f"weights must be either a square array or an 1D array or list with {numvals} elements")
        # Check the input reference
        if ref is None:
            b = np.zeros((numvals,))
        elif type(ref) is list and len(ref) == numvals:
            b = np.asarray(ref)
        elif type(ref) is np.ndarray and ref.shape[0] == numvals and ref.ndim() == 1:
            b = ref
        else:
            raise ValueError(f"If ref is not None, it must be a list or array with {numvals} elements")
        return (W,b)

    def setSolverOption(self, name, value):
        self.snoptions[name] = value

    def useLinearGuess(self):
        """
        Initializes the variables using:
            1. Linear interpolation between initial and final states
            2. Zeros for forces and controls 
            3. Maximum timestep values for timesteps
        """
        # First use the zero guess
        self.useZeroGuess()
        # Now use linear interpolation to set the states
        x_init = np.linspace(self._initial_condition,  self._final_condition,  self.trajopt.num_time_samples)
        self.trajopt.set_initial_guess(xtraj=x_init.transpose())

    def useCustomGuess(self, x_init=None, u_init=None, l_init=None):
        """
        Initialize the decision variables using custom values. Values not given are initialized using useZeroGuess
        """
        self.useZeroGuess()
        self.trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)

    def useZeroGuess(self):
        """
        Initializes the decision variables using:
            1. Zeros for states, controls, and forces
            2. Maximum timestep for timesteps
        """
        x_init = np.zeros(self.trajopt.x.shape)
        u_init = np.zeros(self.trajopt.u.shape)
        l_init = np.zeros(self.trajopt.l.shape)
        t_init = np.ones(self.trajopt.h.shape) * self.trajopt.maximum_timestep
        # Set the initial guess
        self.trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
        # Set the initial guess for the timesteps
        self.trajopt.prog.SetInitialGuess(self.trajopt.h, t_init)

    def solve(self):
        """ Solve the trajectory optimization"""
        solver = SnoptSolver()
        print("Solving trajectory optimization")
        start = timeit.default_timer()
        result = solver.Solve(self.trajopt.prog)
        stop = timeit.default_timer()
        print(f"Elapsed time: {stop-start}")
        # Print a report to the terminal
        utils.printProgramReport(result, self.trajopt.prog)
        return result, stop-start

    def plot(self, result):
        """Plot the results from the optimization"""
        xtraj, utraj, ftraj, _, _ = self.trajopt.reconstruct_all_trajectories(result)
        self.plant.plot_trajectories(xtraj, utraj, ftraj)

    def save(self, result, name="blockopt.pkl"):
        file = "data/slidingblock/" + name
        utils.save(file, self.trajopt.result_to_dict(result))

    def enforceEqualTimesteps(self):
        """ Add a constraint that all timesteps be equal """
        self.trajopt.add_equal_time_constraints()

    def enableDebugging(self, display="terminal"):
        self.trajopt.enable_cost_display(display)

    def finalizeProgram(self):
        """ Finalize the program and add any final options """
        # Set the boundary conditions
        self._set_boundary_conditions()
        # Set the running costs
        self._set_running_costs()
        # Set the solver options
        self._set_snopt_options()

    def _set_snopt_options(self):
        """
        Set the options for SNOPT within the Mathematical Program
        """
        for option in self.snoptions:
            self.trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), option, self.snoptions[option])
    
    def _set_boundary_conditions(self):
        """ Set the boundary conditions in the trajopt """
        self.trajopt.add_state_constraint(knotpoint=0, value=self._initial_condition)
        self.trajopt.add_state_constraint(knotpoint=self.trajopt.num_time_samples-1, value=self._final_condition)
    
    def _set_running_costs(self):
        if self._control_cost is not None:
            R, uref = self._control_cost
            self.trajopt.add_quadratic_running_cost(R, uref, vars=[self.trajopt.u], name="ControlCost")
        if self._state_cost is not None:
            Q, xref = self._state_cost
            self.trajopt.add_quadratic_running_cost(Q, xref, vars=[self.trajopt.x], name="StateCost")

    def generateReport(self, result, elapsed=None, filename=None):
        """ Generate a report for the optimization """
        # Write the Elapsed time to a string
        if elapsed is not None:
            report = f"\nElapsed Time: {elapsed}\n"
        else:
            report = f"\n"
        # Write the SNOPT settings to a string
        report += f"\nSNOPT Settings:\n"
        for option in self.snoptions:
            report += f"\t {option}: {self.snoptions[option]}\n"
        # Write the boundary conditions to a string
        report += f"\nInitial condition: {self._initial_condition}\n"
        report += f"Final condition: {self._final_condition}\n"
        # Write the costs to the string
        if self._control_cost is not None:
            report += f"\nControl Cost:\n"
            report += f"\t weights = {self._control_cost[0]}\n"
            report += f"\t reference = {self._control_cost[1]}\n"
        else:
            report += f"\nControl Cost: none\n"

        if self._state_cost is not None:
            pass
        else:
            report += f"\nState Cost: none\n"
            report += f"\tweights = {self._state_cost[0]}\n"
            report += f"\treference = {self._state_cost[1]}\n"
        
        # Print the report generated from utilities
        utils.printProgramReport(result, prog=self.trajopt.get_program(), filename=filename)
        if filename is None:
            print(report)
        else:
            with open(filename, "a") as file:
                file.write(report)

    def plotConstraints(self, result):
        viewer = ContactConstraintViewer(self.trajopt, self.trajopt.result_to_dict(result))
        viewer.plot_constraints()

    @property
    def plant(self):
        return self.trajopt.plant_f
