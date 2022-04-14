"""
Class for Linear Contact-Implicit MPC

January 26, 2022
Luke Drnach
"""
#TODO: Double check implementation of joint limit linearization in the dynamics
#TODO: Test getTimeIndex when the time is the final value
#TODO: Test _update_contact_linearization works as intended
import numpy as np
import abc

from pydrake.all import MathematicalProgram, SnoptSolver, OsqpSolver
from pydrake.all import PiecewisePolynomial as pp

import pycito.utilities as utils
import pycito.trajopt.constraints as cstr
import pycito.controller.mlcp as mlcp
from pycito.controller.optimization import OptimizationMixin
from pycito.controller.contactestimator import ContactModelEstimator

class _ControllerBase(abc.ABC):
    def __init__(self, plant):
        super().__init__()
        self._plant = plant

    @abc.abstractmethod
    def get_control(self, t, x, u=None):
        raise NotImplementedError

class NullController(_ControllerBase):
    """
    NullController: returns zeros for all state and time pairs
    """
    def __init__(self, plant):
        super(NullController, self).__init__(plant)

    def get_control(self, t, x, u=None):
        return np.zeros((self._plant.multibody.num_actuators(), ))

class OpenLoopController(_ControllerBase):
    """
    OpenLoopController: returns the value of the planned open loop controls
    """
    def __init__(self, plant, time, control):
        super(OpenLoopController, self).__init__(plant)
        self._utraj = pp.ZeroOrderHold(time, control)

    @classmethod
    def fromReferenceTrajectory(cls, reftraj):
        return cls(reftraj.plant, reftraj._time, reftraj._control)

    def get_control(self, t, x, u=None):
        """Open loop control value"""
        t = min(max(self._utraj.start_time(), t), self._utraj.end_time())
        return np.reshape(self._utraj.value(t), (-1,))

class ReferenceTrajectory():
    """
    Container class for holding a optimized trajectory and it's respective plant model
    """
    def __init__(self, plant, time, state, control, force, jointlimit=None):
        self.plant = plant
        self._time = time
        self._state = state
        self._control = control
        nC, nF = plant.num_contacts(), plant.num_friction()
        self._force = force[:nC+nF, :]
        self._slack = force[nC+nF:, :]
        self._jlimit = jointlimit

    @classmethod
    def load(cls, plant, filename):
        data = utils.load(utils.FindResource(filename))
        if data['jointlimit'] is not None:
            return cls(plant, data['time'], data['state'], data['control'], data['force'], data['jointlimit'])
        else:
            return cls(plant, data['time'], data['state'], data['control'], data['force'])

    def getTimeIndex(self, t):
        """Return the index of the last timepoint less than the current time"""
        if t < self._time[0]:
            return 0
        elif t > self._time[-1]:
            return self.num_timesteps
        else:
            return np.argmax(self._time > t) - 1
    
    def getTime(self, index):
        """
        Return the time at the given index, returning the last time if the index is out of bounds
        """
        index = min(max(0,index), self.num_timesteps-1)
        return self._time[index]

    def getState(self, index):
        """
        Return the state at the given index, returning the last state if the index is out of bounds
        """
        index = min(max(0,index), self.num_timesteps-1)
        return self._state[:, index]

    def getControl(self, index):
        """
        Return the control at the given index, returning the last control if the index is out of bounds
        """
        index = min(max(0,index), self.num_timesteps-1)
        return self._control[:, index]

    def getForce(self, index):
        """
        Return the force at the given index, returning the last force vector if the index is out of bounds
        """
        index = min(max(0, index), self.num_timesteps-1)
        return self._force[:, index]

    def getSlack(self, index):
        """
        Return the velocity slacks at the given index, returning the last slack vector if the index is out of bounds
        """
        index = min(max(index,0), self.num_timesteps-1)
        return self._slack[:, index]
    
    def getJointLimit(self, index):
        """
        Return the joint limit forces at the given index, returning the last joint limit vector if the index is out of bounds
        """
        index = min(max(0, index), self.num_timesteps-1)
        return self._jlimit[:, index]

    @property
    def num_timesteps(self):
        return self._time.size

    @property
    def state_dim(self):
        return self._state.shape[0]
    
    @property
    def control_dim(self):
        return self._control.shape[0]

    @property
    def force_dim(self):
        return self._force.shape[0]

    @property
    def slack_dim(self):
        return self._slack.shape[0]

    @property
    def jlimit_dim(self):
        if self.has_joint_limits:
            return self._jlimit.shape[0]
        else:
            return 0

    @property
    def has_joint_limits(self):
        return self._jlimit is not None

class LinearizedContactTrajectory(ReferenceTrajectory):
    def __init__(self, plant, time, state, control, force, jointlimit=None):
        super(LinearizedContactTrajectory, self).__init__(plant, time, state, control, force, jointlimit)           
        # Setup the parameter lists
        self.joint_limit_cstr = None
        self.dynamics_cstr = []
        self.distance_cstr = []
        self.dissipation_cstr = []
        self.friccone_cstr = []
        # Store the linearized parameter values
        self.linearize_trajectory()

    @classmethod
    def load(cls, plant, filename):
        """Class Method for generating a LinearizedContactTrajectory from a file containing a trajectory"""
        data = utils.load(filename)
        if data['jointlimit'] is not None:
            return cls(plant, data['time'], data['state'], data['control'], data['force'], data['jointlimit'])
        else:
            return cls(plant, data['time'], data['state'], data['control'], data['force'])

    def linearize_trajectory(self):
        """Store the linearizations of all the parameters"""
        self._linearize_dynamics()
        if self.has_joint_limits:
            self._linearize_joint_limits()
        # Store the constraint evaluators
        self._distance = cstr.NormalDistanceConstraint(self.plant)
        self._dissipation = cstr.MaximumDissipationConstraint(self.plant)
        self._friccone = cstr.FrictionConeConstraint(self.plant)
        # Linearize the constraints
        self.distance_cstr = [None] * self.num_timesteps
        self.dissipation_cstr = [None] * self.num_timesteps
        self.friccone_cstr = [None] * self.num_timesteps

        for k in range(self.num_timesteps):
            self._linearize_normal_distance(k)
            self._linearize_maximum_dissipation(k)
            self._linearize_friction_cone(k)
        
    def _linearize_dynamics(self):
        """Store the linearization of the dynamics constraints"""
        force_idx = 2*self.state_dim + self.control_dim + 1
        dynamics = cstr.BackwardEulerDynamicsConstraint(self.plant)
        if self.has_joint_limits:
            force = np.concatenate([self._force, self._jlimit], axis=0)
        else:
            force = self._force
        for n in range(self.num_timesteps-1):
            h = np.array([self._time[n+1] - self._time[n]])
            A, _ = dynamics.linearize(h, self._state[:,n], self._state[:, n+1], self._control[:,n], force[:, n+1])
            b =  - A[:, force_idx:].dot(force[:, n+1])
            A = A[:, 1:]
            self.dynamics_cstr.append((A,b))

    def _linearize_normal_distance(self, index):
        """Store the linearizations for the normal distance constraint"""
        A, c = self._distance.linearize(self.getState(index))
        self.distance_cstr[index] = (A, c)

    def _linearize_maximum_dissipation(self, index):
        """Store the linearizations for the maximum dissipation function"""
        state, vslack = self.getState(index), self.getSlack(index)
        A, c = self._dissipation.linearize(state, vslack)
        c -= A[:, state.shape[0]:].dot(vslack)       #Correction term for LCP 
        self.dissipation_cstr[index] = (A, c)

    def _linearize_friction_cone(self, index):
        """Store the linearizations for the friction cone constraint function"""
        state, force = self.getState(index), self.getForce(index)
        A, c = self._friccone.linearize(state, force)
        c -= A[:, state.shape[0]:].dot(force)   #Correction term for LCP
        self.friccone_cstr[index] = (A, c)

    def _linearize_joint_limits(self):
        """Store the linearizations of the joint limits constraint function"""
        jointlimits = cstr.JointLimitConstraint(self.plant)
        self.joint_limit_cstr = []
        nQ = self.plant.multibody.num_positions()
        for x in self._state.transpose():
            A, c = jointlimits.linearize(x[:nQ])
            self.joint_limit_cstr.append((A,c))

    def getDynamicsConstraint(self, index):
        """Returns the linear dynamics constraint at the specified index"""
        index = min(max(0, index), len(self.dynamics_cstr)-1)
        return self.dynamics_cstr[index]

    def getDistanceConstraint(self, index):
        """Returns the normal distance constraint at the specified index"""
        index = min(max(0, index), len(self.distance_cstr)-1)
        return self.distance_cstr[index]

    def getDissipationConstraint(self, index):
        """Returns the maximum dissipation constraint at the specified index"""
        index = min(max(0, index), len(self.dissipation_cstr)-1)
        return self.dissipation_cstr[index]

    def getFrictionConeConstraint(self, index):
        """Returns the friction cone constraint at the specified index"""
        index = min(max(0, index), len(self.friccone_cstr)-1)
        return self.friccone_cstr[index]

    def getJointLimitConstraint(self, index):
        """Returns the joint limit constraint at the specified index"""
        index = min(max(0, index), len(self.joint_limit_cstr)-1)
        return self.joint_limit_cstr[index]

class LinearContactMPC(_ControllerBase, OptimizationMixin):
    def __init__(self, linear_traj, horizon, lcptype=mlcp.CostRelaxedPseudoLinearComplementarityConstraint):
        """
        Plant: a TimesteppingMultibodyPlant instance
        Traj: a LinearizedContactTrajectory
        """
        super().__init__(linear_traj.plant)
        self.lintraj = linear_traj
        self.horizon = horizon
        self.lcp = lcptype
        # Default cost weights
        self._state_weight = np.eye(self.state_dim)
        self._control_weight = np.eye(self.control_dim)
        self._force_weight = np.eye(self.force_dim)
        self._slack_weight = np.eye(self.slack_dim)
        if self.jlimit_dim > 0:
            self._jlimit_weight = np.eye(self.jlimit_dim)
        else:
            self._jlimit_weight = None
        # Default complementarity cost weight
        self._complementarity_weight = 1
        # Use zero or random guess
        self._use_zero_guess = False
        # Set the solver
        self._solver = OsqpSolver()
        self.solveroptions = {}
        self._setup_program()

    def _setup_program(self):
        """
        Seed the MPC program
        
        _setup_program initializes the program with random values in the constraint matrices. The constraints can then be updated by calls to other functions
        """
        # Create holders for the variables, costs, and constraints
        self._clear()
        # Initialize the program with random values
        self._initialize_mpc()        
        for _ in range(self.horizon):
            self._add_decision_variables()
            self._initialize_constraints()
            self._initialize_costs()

    def _clear(self):
        """
        Reset the program and all variables, costs, and constraints to null values
        """
        self._prog = None
        # Variables
        self._dx = []   # States
        self._du = []   # Controls
        self._dl = []   # Forces
        self._ds = []   # Slacks
        self._djl = []
        # Costs
        self._state_cost = []
        self._control_cost = []
        self._force_cost = []
        self._slack_cost = []
        self._limit_cost = []
        # Constraints
        self._dynamics = []
        self._distance = []
        self._dissipation = []
        self._friccone = []
        self._limits = []
        self._initial = []

    def _initialize_mpc(self):
        """
        Initialize the MPC for the current state
        """
        self._prog = MathematicalProgram()
        # Create the initial state and constrain it
        self._dx = [self.prog.NewContinuousVariables(rows = self.state_dim, name='state')]
        dx0 = np.random.default_rng().random((self.state_dim,))
        self.initial = self.prog.AddLinearEqualityConstraint(Aeq = np.eye(self.state_dim), beq=dx0, vars=self._dx[0])
        self.initial.evaluator().set_description('initial state')

    def _add_decision_variables(self):
        """
        Add and initialize decision variables to the program for the current time index
        """
        # Add new variables to the program
        self._dx.append(self.prog.NewContinuousVariables(rows = self.state_dim, name='state'))
        self._du.append(self.prog.NewContinuousVariables(rows = self.control_dim, name='control'))
        self._dl.append(self.prog.NewContinuousVariables(rows = self.force_dim, name='force'))
        self._ds.append(self.prog.NewContinuousVariables(rows = self.slack_dim, name='slack'))
        # Add and initialize joint limits
        if self.jlimit_dim > 0:
            self._djl.append(self.prog.NewContinuousVariables(rows = self.jlimit_dim, name='joint_limits'))

    def _initialize_constraints(self):
        """Initialize the constraints with random values"""
        # Some necessary parameters
        nQ = self.lintraj.plant.multibody.num_positions()
        nV = self.lintraj.plant.multibody.num_velocities()
        nC = self.lintraj.plant.num_contacts()
        nF = self.lintraj.plant.num_friction()
        # Initialize the dynamics constraints
        if self.jlimit_dim > 0:
            dl_all = np.concatenate([self._dl[-1], self._djl[-1]], axis=0)
            self._dynamics.append(cstr.LinearImplicitDynamics.random(2*self._dx[-1].size + self._du[-1].size + dl_all.size, nQ + nV))
            self._dynamics[-1].name = 'linear dynamics'
            self._dynamics[-1].addToProgram(self.prog, self._dx[-2], self._dx[-1], self._du[-1], dl_all)

            self._limits.append(self.lcp.random(nQ, self._djl[-1].size))
            self._limits[-1].name = 'joint limits'
            self._limits[-1].addToProgram(self.prog, self._dx[-1][:nQ], self._djl[-1])
        else:
            self._dynamics.append(cstr.LinearImplicitDynamics.random(2*self._dx[-1].size + self._du[-1].size + self._dl[-1].size, nQ + nV))
            self._dynamics[-1].name = 'linear dynamics'
            self._dynamics[-1].addToProgram(self.prog, self._dx[-2], self._dx[-1], self._du[-1], self._dl[-1])
        # Initialize distance constraint
        self._distance.append(self.lcp.random(nQ+nV, nC))
        self._distance[-1].cost_weight = self._complementarity_weight
        self._distance[-1].name = "distance"
        self._distance[-1].addToProgram(self.prog, self._dx[-1], self._dl[-1][:nC])
        # Initialize dissipation constraint
        self._dissipation.append(self.lcp.random(nQ + nV + nC, nF))
        self._dissipation[-1].cost_weight = self._complementarity_weight
        self._dissipation[-1].name = 'dissipation'
        self._dissipation[-1].addToProgram(self.prog, np.concatenate([self._dx[-1], self._ds[-1]], axis=0), self._dl[-1][nC:nC+nF])
        # Initialize friction cone constraint
        self._friccone.append(self.lcp.random(nQ + nV + nF + nC, nC))
        self._friccone[-1].cost_weight = self._complementarity_weight
        self._friccone[-1].name = 'friction cone'
        self._friccone[-1].addToProgram(self.prog, np.concatenate([self._dx[-1], self._dl[-1][:nC+nF]], axis=0), self._ds[-1])
        
    def _initialize_costs(self):
        """Add cost terms on the most recent variables"""
        # We add quadratic error costs on all variables, regularizing states and controls to 0 and complementarity variables to their trajectory values
        self._state_cost.append(self.prog.AddQuadraticErrorCost(self._state_weight, np.zeros((self.state_dim, 1)), vars=self._dx[-1]))
        self._state_cost[-1].evaluator().set_description('state_cost')
        
        self._control_cost.append(self.prog.AddQuadraticErrorCost(self._control_weight, np.zeros((self.control_dim, 1)), vars=self._du[-1]))
        self._control_cost[-1].evaluator().set_description('control_cost')
    
        self._force_cost.append(self.prog.AddQuadraticErrorCost(self._force_weight, np.ones(self._dl[-1].shape), vars=self._dl[-1]))
        self._force_cost[-1].evaluator().set_description('force_cost')
        
        self._slack_cost.append(self.prog.AddQuadraticErrorCost(self._slack_weight, np.ones(self._ds[-1].shape), vars=self._ds[-1]))
        self._slack_cost[-1].evaluator().set_description('slack_cost')
        
        # Add a cost for joint limits
        if self.jlimit_dim > 0:
            self._limit_cost.append(self.prog.AddQuadraticErrorCost(self._jlimit_weight, np.ones(self._djl[-1].shape), vars=self._djl[-1]))
            self._limit_cost[-1].evaluator().set_description('joint_limit_cost')
        
    def create_mpc_program(self, t, x0):
        """
            Update all the costs and constraints within the MPC program
        
            Arguments: 
                t: (1,) numpy array, time value
                x0: (N,) numpy array, initial state vector
        """
        index = self.lintraj.getTimeIndex(t)
        self._initialize_variables(index)
        self._update_initial_constraint(index, x0)
        self._update_dynamics(index)
        self._update_limits(index+1)
        self._update_contact(index+1)
        self._update_costs(index+1)

    def _initialize_variables(self, index):
        """Set the initial guesses for all variables in the program"""
        #TODO: Better initialization for states and controls
        if self._use_zero_guess:
             self.prog.SetInitialGuess(self.dx, np.zeros((self.state_dim, self.horizon + 1)))
             self.prog.SetInitialGuess(self.du, np.zeros((self.control_dim, self.horizon)))
        else:
            self.prog.SetInitialGuess(self.dx, np.random.default_rng().standard_normal(size = self.dx.shape))
            self.prog.SetInitialGuess(self.du, np.random.default_rng().standard_normal(size = self.du.shape))
        for k, (df, ds) in enumerate(zip(self._dl, self._ds)):
            self.prog.SetInitialGuess(df, self.lintraj.getForce(index + k + 1))
            self.prog.SetInitialGuess(ds, self.lintraj.getSlack(index + k + 1))
        for k, djl in enumerate(self._djl):
            self.prog.SetInitialGuess(djl, self.lintraj.getJointLimit(index + k + 1))
        
    def _update_initial_constraint(self, index, x0):
        """Update the initial state constraint"""
        dx0 = x0 - self.lintraj.getState(index)
        self.initial.evaluator().UpdateCoefficients(np.eye(self.state_dim), dx0)
        self.prog.SetInitialGuess(self._dx[0], dx0)

    def _update_dynamics(self, index):
        """Update the dynamics constraints"""
        for k, dyn in enumerate(self._dynamics):
            A, b = self.lintraj.getDynamicsConstraint(index + k)
            dyn.updateCoefficients(A, b)
                
    def _update_limits(self, index):
        """update the joint limit constraints"""
        for k, limit in enumerate(self._limits):
            A, b = self.lintraj.getJointLimitConstraint(index + k)
            limit.updateCoefficients(A, b)
            limit.initializeSlackVariables()

    def _update_contact(self, index):
        """Update the contact constraints"""
        for k, (dist, diss, fric) in enumerate(zip(self._distance, self._dissipation, self._friccone)):
            A, b = self.lintraj.getDistanceConstraint(index + k)
            dist.updateCoefficients(A, b)
            dist.initializeSlackVariables()
            A, b = self.lintraj.getDissipationConstraint(index + k)
            diss.updateCoefficients(A, b)
            diss.initializeSlackVariables()
            A, b = self.lintraj.getFrictionConeConstraint(index + k)
            fric.updateCoefficients(A, b)
            fric.initializeSlackVariables()

    def _update_costs(self, index):
        """Update the quadratic cost values"""
        for xcost, ucost in zip(self._state_cost, self._control_cost):
            xcost.evaluator().UpdateCoefficients(2 * self._state_weight, np.zeros((self.state_dim,)), np.zeros((1,)))
            ucost.evaluator().UpdateCoefficients(2 * self._control_weight, np.zeros((self.control_dim)), np.zeros((1,)))
        for k, (fcost, scost) in enumerate(zip(self._force_cost, self._slack_cost)):
            f_ref = self.lintraj.getForce(index + k)
            fcost.evaluator().UpdateCoefficients(2*self._force_weight,
                                     -2*self._force_weight.dot(f_ref), f_ref.dot(self._force_weight.dot(f_ref)))
            s_ref = self.lintraj.getSlack(index + k)
            scost.evaluator().UpdateCoefficients(2 * self._slack_weight, 
                                    -2 * self._slack_weight.dot(s_ref),
                                    s_ref.dot(self._slack_weight.dot(s_ref)))
        for k, jcost in enumerate(self._limit_cost):
            j_ref = self.lintraj.getJointLimit(index + k)
            jcost.evaluator().UpdateCoefficients(2 * self._jlimit_weight, 
                                    -2 * self._jlimit_weight.dot(j_ref),
                                    j_ref.dot(self._jlimit_weight.dot(j_ref)))

    def get_control(self, t, x, u=None):
        """"
        Return the MPC feedback controller
        Thin wrapper for do_mpc
        """
        return self.do_mpc(t, x)

    def do_mpc(self, t, x0):
        """
        Solve the MPC problem and return the updated control signal
        """
        print(f"Creating MPC at time {t:0.3f}")
        self.create_mpc_program(t, x0)
        print(f"Solving MPC at time {t:0.3f}")
        result = self.solve()
        index = self.lintraj.getTimeIndex(t)
        u = self.lintraj.getControl(index)
        if result.is_success():
            print(f"MPC succeeded at time {t:0.3f}")
            du = result.GetSolution(self._du[0])
            return u + du
        else:
            if result.get_solver_id().name() == "SNOPT/fortran":
                insert = f'with exit code {result.get_solver_details().info}'    
            elif result.get_solver_id().name() == 'OSQP':
                insert = f'with exit code {result.get_solver_details().status_val}'
            else:
                insert = ''
            print(f"MPC failed at time {t:0.3f} using {result.get_solver_id().name()} {insert}. Returning open loop control")
            return u

    @property
    def state_dim(self):
        return self.lintraj.state_dim

    @property
    def control_dim(self):
        return self.lintraj.control_dim

    @property
    def force_dim(self):
        return self.lintraj.force_dim

    @property
    def slack_dim(self):
        return self.lintraj.slack_dim

    @property
    def jlimit_dim(self):
        return self.lintraj.jlimit_dim

    @property
    def statecost(self):
        return self._state_weight

    @statecost.setter
    def statecost(self, val):
        assert val.shape == self._state_weight.shape, f"statecost must be an {self._state_weight.shape} array"
        self._state_weight = val

    @property
    def controlcost(self):
        return self._control_weight

    @controlcost.setter
    def controlcost(self, val):
        assert val.shape == self._control_weight.shape, f"controlcost must be an {self._control_weight.shape} array"
        self._control_weight = val

    @property
    def forcecost(self):
        return self._force_weight

    @forcecost.setter
    def forcecost(self, val):
        assert val.shape == self._force_weight.shape, f"forcecost must be an {self._force_weight.shape} array"
        self._force_weight = val

    @property
    def slackcost(self):
        return self._slack_weight

    @slackcost.setter
    def slackcost(self, val):
        assert val.shape == self._slack_weight.shape, f"slackcost  must be a {self._slack_weight.shape} array"
        self._slack_weight = val

    @property
    def limitcost(self):
        return self._jlimit_weight

    @limitcost.setter
    def limitcost(self, val):
        assert self.jlimit_dim > 0, f"Current model has no joint limits"
        assert val.shape == self._jlimit_weight.shape, f"limitcost must be a {self._jlimit_weight.shape} array"
        self._jlimit_weight = val

    @property
    def complementaritycost(self):
        return self._complementarity_weight

    @complementaritycost.setter
    def complementaritycost(self, val):
        assert isinstance(val, (int, float)), f"complementaritycost must be a float or an int"
        self._complementarity_weight = val

    @property
    def dx(self):
        """State error getter"""
        if self._dx is not []:
            return np.column_stack(self._dx)
        else:
            return None

    @property
    def du(self):
        """Control error getter"""
        if self._du is not []:
            return np.column_stack(self._du)
        else:
            return None

    @property
    def dl(self):
        """Force error getter"""
        if self._dl is not []:
            return np.column_stack(self._dl)
        else:
            return None

    @property
    def ds(self):
        """Slack error getter"""
        if self._ds is not []:
            return np.column_stack(self._ds)
        else:
            return None

    @property
    def djl(self):
        """Joint limit error getter"""
        if self._djl is not [] and self.jlimit_dim > 0:
            return np.column_stack(self._djl)
        else:
            return None

    def use_zero_guess(self):
        self._use_zero_guess = True

    def use_random_guess(self):
        self._use_zero_guess = False

class ContactAdaptiveMPC(LinearContactMPC):
    def __init__(self, estimator, linear_traj, horizon, lcptype=mlcp.CostRelaxedPseudoLinearComplementarityConstraint):
        super().__init__(linear_traj, horizon, lcptype)
        assert isinstance(estimator, ContactModelEstimator), f"estimator must be an instance of ContactModelEstimator"
        self.estimator = estimator

    def enableLogging(self):
        """Enable solution logging for both the estimator and the controller"""
        super().enableLogging()
        self.estimator.enableLogging()
        
    def disableLogging(self):
        """Disable solution logging for both the estimator and the controller"""
        super().disableLoggin()
        self.estimator.disableLogging()

    def getControllerLogs(self):
        """Return the predictive controller solution logs"""
        return self.logger

    def getEstimatorLogs(self):
        """Return the contact estimator solution logs"""
        return self.estimator.logger
        
    def get_control(self, t, x, u):
        """
            Get a new control based on the previous control
        
            Arguments:
                t: (1, ) numpy array, the current time
                x: (N, ) numpy array, the current state
                u_old: (M, ) numpy array, the previous control
            
            Return values:
                u: (M, ) the updated current control
        """    
        model = self.estimator.estimate_contact(t, x, u)
        self._update_contact_linearization(t, model)
        return self.do_mpc(t, x)

    def _update_contact_linearization(self, t, model):
        """
            Update the contact constraints linearization used in MPC
        """
        self.lintraj.plant.terrain = model
        index = self.lintraj.getTimeIndex(t)
        for k in range(self.horizon):
            self.lintraj._linearize_normal_distance(index + k)
            self.lintraj._linearize_friction_cone(index + k)

    def getContactEstimationTrajectory(self):
        return self.estimator.traj

if __name__ == "__main__":
    print("Hello from MPC!")