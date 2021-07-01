"""
General methods for creating optimizers for multibody systems

Luke Drnach
June 28, 2021
"""
#TODO: Make a configuration class to hold standard details, like cost weights, boundary constraints, SnoptOptions, etc. Make the class saveable and loadable - DONE
#TODO: Create special instances for A1 and sliding block, with routines for static, lifting, and walking for A1

#TODO: Write scripts to create configurations
#TODO: Write scripts to load a sequence of configurations, run the optimization, and save the output to a subdirectory (optimization problem/date/run number)
#TODO: To the output directory, save the configuration, the trajectory data, the figures, and the trajectory optimization report


import numpy as np
import matplotlib.pyplot as plt
import abc, os
import pickle as pkl
# Custom imports
from trajopt import contactimplicit as ci
import utilities as utils
import decorators as deco
from systems.A1.a1 import A1VirtualBase
from systems.block.block import Block
# Drake imports
from pydrake.all import PiecewisePolynomial

def create_guess_from_data(time, data, num_samples):
    """Resample data to create a new initial guess for optimization. Assume piecewise linear between datapoints for sampling"""
    traj = PiecewisePolynomial.FirstOrderHold(time, data)
    new_time = np.linspace(0, traj.end_time(), num_samples)
    return traj.vector_values(new_time)

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
    """Boilerplate class for handling common trajectory optimization requests """
    def __init__(self, options=None, min_time=1, max_time=1, num_time_samples=101):
        """ Initialize the optimization problem"""
        # Create a plant
        self.plant = self.make_plant()
        self.options = self.defaultOptimizationOptions()
        # Create the trajectory optimization problem
        if options is None:
            options = self.defaultOptimizationOptions()
        self.trajopt = ci.ContactImplicitDirectTranscription(self.plant, 
                                self.plant.multibody.CreateDefaultContext(),
                                num_time_samples = num_time_samples,
                                minimum_timestep = min_time/(num_time_samples-1),
                                maximum_timestep = max_time/(num_time_samples-1),
                                options=options)
        # Set solver options
        self.trajopt.setSolverOptions("Iterations"
        ) = {"Iterations Limit":10000,
                                "Major Feasibility Tolerance": 1e-6,
                                "Major Optimality Tolerance": 1e-6,
                                "Scale Option": 2}
        # Default initial and final conditions
        self._initial_condition = None
        self._final_condition = None
        # Default control and state weights
        self._control_cost = None
        self._state_cost = None
        # Flag for debugging
        self.debugging_enabled = False

    @abc.abstractmethod
    def make_plant(self):
        """Returns a finalized timestepping multibody plant system"""
        raise NotImplementedError

    @classmethod
    def loadConfig(cls, file):
        config = OptimizationConfiguration.load(file)

    @staticmethod
    def defaultOptimizationOptions():
        return ci.OptimizationOptions()    

    @staticmethod
    def _check_quadratic_cost_weights(weights, ref, numvals):
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

    def useLinearGuess(self):
        """
        Initializes the variables using:
            1. Linear interpolation between initial and final states
            2. Use a constant control if a control reference is provided in the cost, otherwise use zeros
            2. Use zeros for the forces
            3. Maximum timestep values for timesteps
        """
        # First use the zero guess
        self.useZeroGuess()
        # Now use linear interpolation to set the states
        if self._initial_condition is None or self._final_condition is None:
            raise RuntimeError("Boundary conditions must be set to use a linear guess")
        x_init = np.linspace(self._initial_condition,  self._final_condition,  self.trajopt.num_time_samples)
        # Check if there is a reference for the controls
        if self._control_cost is not None:
            R, ref = self._control_cost
            u_init = np.linspace(ref, ref, self.trajopt.num_time_samples).transpose()
            self.trajopt.set_initial_guess(xtraj=x_init, utraj=u_init)
        else:
            self.trajopt.set_initial_guess(xtraj=x_init.transpose())

    def useCustomGuess(self, x_init=None, u_init=None, l_init=None):
        """
        Initialize the decision variables using custom values. Values not given are initialized using useZeroGuess
        """
        self.useZeroGuess()
        self.trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)

    def useGuessFromFile(self, filename):
        data = utils.load(filename)
        # Re-sample to create the initial guess
        x_init = create_guess_from_data(data['time'], data['state'], self.trajopt.num_time_samples)
        u_init = create_guess_from_data(data['time'], data['state'], self.trajopt.num_time_samples)
        l_init = create_guess_from_data(data['time'], data['state'], self.trajopt.num_time_samples)
        # Add the guess to the program
        self.useCustomGuess(x_init=x_init, u_init=u_init, l_init=l_init)

    def enforceEqualTimesteps(self):
        """ Add a constraint that all timesteps be equal """
        self.trajopt.add_equal_time_constraints()

    def enableDebugging(self, display="terminal"):
        self.debugging_enabled = True
        self.trajopt.enable_cost_display(display)

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
    
    def setBoundaryConditions(self, initial=None, final=None):
        """ set the values for the boundary conditions """
        nx = self.plant.multibody.num_positions() + self.plant.multibody.num_velocities()
        if type(initial) is list and len(initial) == nx:
            self._initial_condition = np.array(initial)
        elif type(initial) is np.ndarray and initial.shape[0] == nx:
            self._initial_condition = initial
        else:
            raise ValueError(f"initial must be an array or list with {nx} elements")
        if type(final) is list and len(final) == nx:
            self._final_condition = np.array(final)
        elif type(final) is np.ndarray and final.shape[0] == nx:
            self._final_condition = final
        else:
            raise ValueError(f"final must be an array or list with {nx} elements")

    def useFinalStateCost(self, weight = None):
        self.final_state_cost = weight

    def useFinalStateConstraint(self):
        self.final_state_cost = None

    def useFinalTimeCost(self, weight=1):
        self.final_time_cost = 1

    def finalizeProgram(self):
        """Add the state constraints and costs to the program, and set the solver options"""
        # Set boundary conditions
        self._set_boundary_conditions()
        # Set cost weights
        self._set_running_costs()
        # Set Solver options
        self.trajopt.setSolverOptions(**self.solver_options)

    def solve(self):
        return self.trajopt.solve()

    def plot(self, result, show=True, savename=None):
        """Plot the results of optimization"""
        xtraj, utraj, ftraj, *_, = self.trajopt.reconstruct_all_trajectories(result)
        self.plant.plot_trajectories(xtraj, utraj, ftraj, show, savename)

    def plotConstraints(self, result, show=True, savename=None):
        """Plot the constraints and dual solutions"""
        viewer = ci.ContactConstraintViewer(self.trajopt, self.trajopt.result_to_dict(result))
        viewer.plot_constraints(show, savename)

    def saveDebugFigure(self, savename='CostsAndConstraints.png'):
        if self.debugging_enabled:
            self.trajopt.printer.save_and_close(savename)

    def saveResults(self, result, name="trajoptresults.pkl"):
        file = "data/" + name
        utils.save(file, self.trajopt.result_to_dict(result))

    def saveReport(self, result=None, savename=None):
        text = self.trajopt.generate_report(result)
        if savename is not None:
            dir = os.path.dirname(savename)
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(savename, "w") as file:
                file.write(text)
    
    def _set_boundary_conditions(self):
        """ Set the boundary conditions in the trajopt """
        self.trajopt.add_state_constraint(knotpoint=0, value=self._initial_condition)
        if self.final_state_cost is None:
            # Use a final state constraint
            self.trajopt.add_state_constraint(knotpoint=self.trajopt.num_time_samples-1, value=self._final_condition)
        else:
            # Use a quadratic final state cost
            self.trajopt.add_final_cost(self._final_state_cost, vars=[self.trajopt.x[:,-1]], name="FinalStateCost")            
    
    def _set_running_costs(self):
        """Set the running costs for the controls and the states"""
        if self._control_cost is not None:
            R, uref = self._control_cost
            self.trajopt.add_quadratic_running_cost(R, uref, vars=[self.trajopt.u], name="ControlCost")
        if self._state_cost is not None:
            Q, xref = self._state_cost
            self.trajopt.add_quadratic_running_cost(Q, xref, vars=[self.trajopt.x], name="StateCost")

class BlockOptimizer(SystemOptimizer):
    @staticmethod
    def make_system():
        plant = Block()
        plant.Finalize()
        return plant

class A1VirtualBaseOptimizer(SystemOptimizer):
    #TODO: Add joint limits to initial guess methods
    @staticmethod
    def make_system():
        a1 = A1VirtualBase()
        a1.terrain.friction = 1.0
        a1.Finalize()
        return a1