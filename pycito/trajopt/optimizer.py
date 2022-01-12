
"""
General methods for creating optimizers for multibody systems

Luke Drnach
June 28, 2021
"""

#TODO: In general, initialize slack variable trajectories, timesteps
#TODO: Debug this entire file framework (HARD)

import numpy as np
import abc, os
import pickle as pkl
# Custom imports
from pycito.trajopt import contactimplicit as ci
import pycito.utilities as utils
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.systems.block.block import Block
# Drake imports
from pydrake.all import PiecewisePolynomial

def create_guess_from_data(time, data, num_samples):
    """Resample data to create a new initial guess for optimization. Assume piecewise linear between datapoints for sampling"""
    traj = PiecewisePolynomial.FirstOrderHold(time, data)
    new_time = np.linspace(0, traj.end_time(), num_samples)
    return traj.vector_values(new_time)

#TODO: (Minor) add setter methods to input check the optimization configurations
class OptimizerConfiguration():
    def __init__(self):
        """Set the default configuration variables for trajectory optimization"""
        # Trajectory optimization settings
        self.num_time_samples = 101
        self.maximum_time = 1
        self.minimum_time = 1
        # Complementarity settings
        self.complementarity = 'useNonlinearComplementarityWithConstantSlack'
        self.complementarity_cost_weight = None
        self.complementarity_slack = 0
        # Boundary Conditions
        self.initial_state = None
        self.final_state = None
        # Control and State Cost Weights
        self.quadratic_control_cost = None
        self.quadratic_state_cost = None
        # Final cost weights
        self.final_time_cost = None
        self.final_state_cost = None
        # Initial guess type
        self.initial_guess = 'useZeroGuess'
        # Solver options
        self.solver_options = {}
        #Other
        self.useFixedTimesteps = False
        self.useCostDisplay = None

    @classmethod
    def load(cls, filename=None):
        """Load a configuration file from disk"""
        # Check that the file exists
        filename = utils.FindResource(filename)
        # Load the configuration data from the file
        with open(filename, 'rb') as input:
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
        with open(filename, 'wb') as output:
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
        self.trajopt.setSolverOptions({"Iterations limit":10000,
                                "Major feasibility tolerance": 1e-6,
                                "Major optimality tolerance": 1e-6,
                                "Scale option": 2})
        # Default initial and final conditions
        self._initial_condition = None
        self._final_condition = None
        # Default control and state weights
        self._control_cost = None
        self._state_cost = None
        self.final_state_cost = None
        self.final_time_cost = None
        # Flag for debugging
        self.debugging_enabled = False
        # Initialization string
        self.initialization_string = ""

    @abc.abstractmethod
    def make_plant(self):
        """Returns a finalized timestepping multibody plant system"""
        raise NotImplementedError

    @classmethod
    def buildFromConfig(cls, configuration):
        if isinstance(configuration, OptimizerConfiguration):
            config = configuration        
        elif isinstance(configuration, str) and os.path.exists(configuration):
            config = OptimizerConfiguration.load(configuration)
        else:
            raise RuntimeError(f"config must be either an OptimizerConfiguration object or a valid filename")
        # Check the complementarity option
        options = ci.OptimizationOptions()
        if hasattr(options, config.complementarity):
            eval(f"options.{config.complementarity}()")
        else:
            raise RuntimeError(f"{config.complementarity} is not an attribute of {type(options).__name__}")
        # Create the optimzer
        optimizer = cls(options, config.minimum_time, config.maximum_time, config.num_time_samples)
        # Update the complementarity information from the configuration file
        if config.complementarity_cost_weight is not None:
            optimizer.trajopt.complementarity_cost_weight = config.complementarity_cost_weight
        if config.complementarity_slack is not None:
            optimizer.trajopt.slack = config.complementarity_slack
        # Add boundary conditions
        optimizer.setBoundaryConditions(initial = config.initial_state, final=config.final_state)
        # Set final cost weight, if desired
        if config.final_state_cost is not None:
            optimizer.useFinalStateCost(config.final_state_cost)
        # Add final time cost, if desired
        if config.final_time_cost is not None:
            optimizer.useFinalTimeCost(config.final_time_cost)
        # Use fixed timesteps
        if config.useFixedTimesteps:
            optimizer.enforceEqualTimesteps()
        # Add control cost if desired
        if config.quadratic_control_cost is not None:
            R, ref = config.quadratic_control_cost
            optimizer.setControlWeights(R, ref)
        # Add state cost if desired
        if config.quadratic_state_cost is not None:
            Q, xref = config.quadratic_state_cost
            optimizer.setStateWeights(Q, xref)
        # Add cost display
        if config.useCostDisplay is not None:
            optimizer.enableDebugging(config.useCostDisplay)
        # Update the solver options
        if config.solver_options is not {}:
            optimizer.trajopt.setSolverOptions(config.solver_options)
        # Set the initial guess
        if isinstance(config.initial_guess, str):
            if hasattr(optimizer, config.initial_guess):
                eval(f"optimizer.{config.initial_guess}()")
            else:
                raise AttributeError(f"{type(optimizer).__name__} has no attribute {config.initial_guess}")
        elif isinstance(config.initial_guess[0], str):
            if hasattr(optimizer, config.initial_guess[0]):
                fcnstr, *args, = config.initial_guess
                eval(f"optimizer.{fcnstr}(*args)")
            else:
                raise AttributeError(f"{type(optimizer).__name__} has no attribute {config.initial_guess[0]}")
        else:
            optimizer.useZeroGuess()

        return optimizer

    @staticmethod
    def defaultOptimizationOptions():
        return ci.OptimizationOptions()    

    @staticmethod
    def _check_quadratic_cost_weights(weights, ref, numvals):
        if type(weights) is list and len(weights) == numvals:
            W = np.diag(weights)
        elif type(weights) is np.ndarray and weights.shape[0] == numvals:
            if weights.ndim == 1:
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
        elif type(ref) is np.ndarray and ref.shape[0] == numvals and ref.ndim == 1:
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
        self.initialization_string = "Zero Guess"

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
        x_init = np.linspace(self._initial_condition,  self._final_condition,  self.trajopt.num_time_samples).transpose()
        # Check if there is a reference for the controls
        if self._control_cost is not None:
            R, ref = self._control_cost
            u_init = np.linspace(ref, ref, self.trajopt.num_time_samples).transpose()
            self.trajopt.set_initial_guess(xtraj=x_init, utraj=u_init)
        else:
            self.trajopt.set_initial_guess(xtraj=x_init)
        self.initialization_string = "Linear Guess"

    def useCustomGuess(self, x_init=None, u_init=None, l_init=None):
        """
        Initialize the decision variables using custom values. Values not given are initialized using useZeroGuess
        """
        self.useZeroGuess()
        self.trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init)
        self.initialization_string = "Custom Guess"

    def useGuessFromFile(self, filename):
        data = utils.load(filename)
        # Re-sample to create the initial guess
        x_init = create_guess_from_data(data['time'], data['state'], self.trajopt.num_time_samples)
        u_init = create_guess_from_data(data['time'], data['state'], self.trajopt.num_time_samples)
        l_init = create_guess_from_data(data['time'], data['state'], self.trajopt.num_time_samples)
        # Add the guess to the program
        self.useCustomGuess(x_init=x_init, u_init=u_init, l_init=l_init)
        self.initialization_string = f"{filename}"

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
    
    def setSolverOptions(self, options_dict={}):
        self.trajopt.setSolverOptions(options_dict)

    def solve(self):
        return self.trajopt.solve()

    def plot(self, result, show=True, savename=None):
        """Plot the results of optimization"""
        xtraj, utraj, ftraj, *_, = self.trajopt.reconstruct_all_trajectories(result)
        self.plant.plot_trajectories(xtraj, utraj, ftraj, show, savename)

    def plotConstraints(self, result, show=True, savename=None):
        """Plot the constraints and dual solutions"""
        if isinstance(result, dict):
            viewer = ci.ContactConstraintViewer(self.trajopt, result)
        else:
            viewer = ci.ContactConstraintViewer(self.trajopt, self.trajopt.result_to_dict(result))
        viewer.plot_constraints(show=show, savename=savename)

    def saveDebugFigure(self, savename='CostsAndConstraints.png'):
        if self.debugging_enabled:
            self.trajopt.printer.save_and_close(savename)

    def saveResults(self, result, name="trajoptresults.pkl"):
        if isinstance(result, dict):
            utils.save(name, result)
        else:
            utils.save(name, self.trajopt.result_to_dict(result))

    def saveReport(self, result=None, savename=None):
        text = self.trajopt.generate_report(result)
        text += f"\nInitialization: {self.initialization_string}"
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
    def make_plant():
        plant = Block()
        plant.Finalize()
        return plant

class BlockOptimizerConfiguration(OptimizerConfiguration):
    @classmethod
    def defaultBlockConfig(cls):
        """Return the default optimization configuration for A1 static standing optimization"""
        config = cls()
        block = Block()
        block.Finalize()
        # Discretization parameters
        config.num_time_samples = 101
        config.maximum_time = 1
        config.minimum_time = 1
        # Complementarity parameters
        config.complementarity = 'useNonlinearComplementarityWithConstantSlack'
        config.complementarity_cost_weight = None
        config.complementarity_slack = 0.
        # State constraints
        config.initial_state = np.array([0., 0.5, 0., 0.])
        config.final_state = np.array([5.0, 0.5, 0., 0.])
        # Cost weights
        R = 10 * np.eye(1)
        uref = np.zeros((1,))
        config.quadratic_control_cost = (R, uref)
        Q = np.eye(4)
        config.quadratic_state_cost = (Q, config.final_state)
        # Set the initial guess type
        config.initial_guess = 'useLinearGuess'
        # Solver options
        config.solver_options = {"Iterations limit": 10000,
                                "Major feasibility tolerance": 1e-6,
                                "Major optimality tolerance": 1e-6,
                                "Scale option": 2}
        # Enforce equal timesteps
        config.useFixedTimesteps = True
        # Enable cost display
        config.useCostDisplay = 'figure'
        # Return the configuration
        return config

class A1VirtualBaseOptimizer(SystemOptimizer):
    #TODO: Add joint limits to initial guess methods
    @staticmethod
    def make_plant():
        a1 = A1VirtualBase()
        a1.terrain.friction = 1.0
        a1.Finalize()
        return a1
    
    def useZeroGuess(self):
        """
        Initializes the decision variables using:
            1. Zeros for states, controls, and forces
            2. Maximum timestep for timesteps
        """
        x_init = np.zeros(self.trajopt.x.shape)
        u_init = np.zeros(self.trajopt.u.shape)
        l_init = np.zeros(self.trajopt.l.shape)
        jl_init = np.zeros(self.trajopt.jl.shape)
        t_init = np.ones(self.trajopt.h.shape) * self.trajopt.maximum_timestep
        # Set the initial guess
        self.trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init, jltraj=jl_init)
        # Set the initial guess for the timesteps
        self.trajopt.prog.SetInitialGuess(self.trajopt.h, t_init)
        self.initialization_string = "Zero Guess"

    def useCustomGuess(self, x_init=None, u_init=None, l_init=None, jl_init=None):
        """
        Initialize the decision variables using custom values. Values not given are initialized using useZeroGuess
        """
        self.useZeroGuess()
        self.trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init, jltraj=jl_init)
        self.initialization_string = "Custom Guess"

    def useGuessFromFile(self, filename):
        data = utils.load(filename)
        # Re-sample to create the initial guess
        x_init = create_guess_from_data(data['time'], data['state'], self.trajopt.num_time_samples)
        u_init = create_guess_from_data(data['time'], data['control'], self.trajopt.num_time_samples)
        l_init = create_guess_from_data(data['time'], data['force'], self.trajopt.num_time_samples)
        jl_init = create_guess_from_data(data['time'], data['jointlimit'], self.trajopt.num_time_samples)
        # Add the guess to the program
        self.useCustomGuess(x_init=x_init, u_init=u_init, l_init=l_init, jl_init=jl_init)
        self.initialization_string = f"{filename}"

    def plot(self, result, show=True, savename=None):
        """Plot the results of optimization"""
        xtraj, utraj, ftraj, jltraj, *_, = self.trajopt.reconstruct_all_trajectories(result)
        self.plant.plot_trajectories(xtraj, utraj, ftraj, jltraj, show, savename)

class A1OptimizerConfiguration(OptimizerConfiguration):
    @classmethod
    def defaultStandingConfig(cls):
        """Return the default optimization configuration for A1 static standing optimization"""
        config = cls()
        a1 = A1VirtualBase()
        a1.Finalize()
        # Discretization parameters
        config.num_time_samples = 21
        config.maximum_time = 2
        config.minimum_time = 2
        # Complementarity parameters
        config.complementarity = 'useNonlinearComplementarityWithCost'
        config.complementarity_cost_weight = 1
        config.complementarity_slack = None
        # State constraints
        pose = a1.standing_pose()
        no_vel = np.zeros((a1.multibody.num_velocities(), ))
        x0 = np.concatenate((pose, no_vel), axis=0)
        xf = x0.copy()
        config.initial_state = x0
        config.final_state = xf 
        # Cost weights
        uref, _  = a1.static_controller(qref = pose)
        R = 0.01 * np.eye(uref.shape[0])
        config.quadratic_control_cost = (R, uref)
        # Set the initial guess type
        config.initial_guess = 'useLinearGuess'
        # Solver options
        config.solver_options = {"Iterations limit": 1000000,
                                "Major feasibility tolerance": 1e-6,
                                "Major optimality tolerance": 1e-6,
                                "Scale option": 2}
        # Enable cost display
        config.useCostDisplay = 'figure'
        # Return the configuration
        return config
    
    @classmethod
    def defaultLiftingConfig(cls):
        config = cls.defaultStandingConfig()
        a1 = A1VirtualBase()
        a1.Finalize()
        # Update the discretization
        config.minimum_time = 1
        # Update the boundary constraints
        pose2 = config.initial_state[:a1.multibody.num_positions()].copy()
        pose2[2] = pose2[2]/2
        # Solve for a feasible pose
        pose2_ik, _ = a1.standing_pose_ik(pose2[:6], guess=pose2.copy())
        config.initial_state[:a1.multibody.num_positions()] = pose2_ik
        # Update the state cost
        Q = 10*np.eye(config.final_state.shape[0])
        config.quadratic_state_cost = (Q, config.final_state)
        # Return the configuration
        return config

    @classmethod
    def defaultWalkingConfig(cls):
        """Default optimization configuration for a1 walking"""
        config = cls.defaultStandingConfig()
        # Use fewer major iterations to control the total time
        config.solver_options['Major iterations limit'] = 5000 
        # Update the discretization
        config.minimum_time = 0.5
        # Update the final constraint
        xf = config.initial_state.copy()
        xf[0] = 1.
        config.final_state = xf
        # Add state cost
        Q = 10*np.eye(config.final_state.shape[0])
        config.quadratic_state_cost = (Q, config.final_state)
        # Return 
        return config

if __name__ == "__main__":
    print('Hello from optimizer.py!')