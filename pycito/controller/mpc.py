"""
Class for Linear Contact-Implicit MPC

January 26, 2022
Luke Drnach
"""
#TODO: Allow user to specify MPC solver, settings
#TODO: Double check implementation of joint limit linearization in the dynamics

import numpy as np

from pydrake.all import MathematicalProgram, Solve

import pycito.utilities as utils
import pycito.trajopt.constraints as cstr
import pycito.controller.mlcp as mlcp

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
    def __init__(self, plant, time, state, control, force, jointlimit=None, lcp=mlcp.CostRelaxedMixedLinearComplementarity):
        super(LinearizedContactTrajectory, self).__init__(plant, time, state, control, force, jointlimit)
        self.lcp = lcp
        # Store the trajectory values
        if self.has_joint_limits:
            self._linearize_joint_limits()
        else:
            self.joint_limit_cstr = None
        # Store the linearized parameter values
        self._linearize_dynamics()
        self._linearize_normal_distance()
        self._linearize_maximum_dissipation()
        self._linearize_friction_cone()

    @classmethod
    def load(cls, plant, filename, lcp=mlcp.CostRelaxedMixedLinearComplementarity):
        """Class Method for generating a LinearizedContactTrajectory from a file containing a trajectory"""
        data = utils.load(filename)
        if data['jointlimit'] is not None:
            return cls(plant, data['time'], data['state'], data['control'], data['force'], data['jointlimit'], lcp=lcp)
        else:
            return cls(plant, data['time'], data['state'], data['control'], data['force'], lcp=lcp)

    def _linearize_dynamics(self):
        """Store the linearization of the dynamics constraints"""
        self.dynamics_cstr = []
        force_idx = 2*self.state_dim + self.control_dim + 1
        dynamics = cstr.BackwardEulerDynamicsConstraint(self.plant)
        if self.has_joint_limits:
            force = np.concatenate([self._force, self._jlimit], axis=0)
        else:
            force = self._force
        for n in range(self.num_timesteps-1):
            h = np.array([self._time[n+1] - self._time[n]])
            A, _ = dynamics.linearize(h, self._state[:,n], self._state[:, n+1], self._control[:,n+1], force[:, n+1])
            b = A[:,0] - A[:, force_idx:].dot(force[:, n+1])
            A = A[:, 1:]
            self.dynamics_cstr.append(cstr.LinearImplicitDynamics(A, b))

    def _linearize_normal_distance(self):
        """Store the linearizations for the normal distance constraint"""
        distance = cstr.NormalDistanceConstraint(self.plant)
        self.distance_cstr = []
        B = np.zeros((self.plant.num_contacts(), self.plant.num_contacts()))
        for x in self._state.transpose():
            A, c = distance.linearize(x)
            self.distance_cstr.append(self.lcp(A, B, c))

    def _linearize_maximum_dissipation(self):
        """Store the linearizations for the maximum dissipation function"""
        dissipation = cstr.MaximumDissipationConstraint(self.plant)
        self.dissipation_cstr = []
        B = np.zeros((self.plant.num_friction(), self.plant.num_friction()))
        for x, s in zip(self._state.transpose(), self._slack.transpose()):
            A, c = dissipation.linearize(x, s)
            c[:, 0] -= A[:, x.shape[0]:].dot(s)       #Correction term for LCP 
            self.dissipation_cstr.append(self.lcp(A, B, c))

    def _linearize_friction_cone(self):
        """Store the linearizations for the friction cone constraint function"""
        friccone = cstr.FrictionConeConstraint(self.plant)
        self.friccone_cstr = []
        B = np.zeros((self.plant.num_contacts(), self.plant.num_contacts()))
        for x, f in zip(self._state.transpose(), self._force.transpose()):
            A, c = friccone.linearize(x, f)
            c[:, 0] -= A[:, x.shape[0]:].dot(f)   #Correction term for LCP
            self.friccone_cstr.append(self.lcp(A, B, c))

    def _linearize_joint_limits(self):
        """Store the linearizations of the joint limits constraint function"""
        jointlimits = cstr.JointLimitConstraint(self.plant)
        self.joint_limit_cstr = []
        B = np.zeros((2*jointlimits.num_joint_limits, 2*jointlimits.num_joint_limits))
        nQ = self.plant.multibody.num_positions()
        for x in self._state.transpose():
            A, c = jointlimits.linearize(x[:nQ])
            self.joint_limit_cstr.append(self.lcp(A, B, c))
    
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

class LinearContactMPC():
    def __init__(self, linear_traj, horizon):
        """
        Plant: a TimesteppingMultibodyPlant instance
        Traj: a LinearizedContactTrajectory
        """
        self.lintraj = linear_traj
        self.horizon = horizon
        # Internal variables
        self.prog = None
        self._dx = []   # States
        self._du = []   # Controls
        self._dl = []   # Forces
        self._ds = []   # Slacks
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

    def do_mpc(self, t, x0):
        """
        Solve the MPC problem and return the updated control signal
        """
        self.create_mpc_program(t, x0)
        result = self.solve()
        du = result.GetSolution(self._du[:, 0])
        index = self.lintraj.GetTimeIndex(t)
        u = self.lintraj.GetControl(index)
        return u + du

    def create_mpc_program(self, t, x0):
        """
        Create a MathematicalProgram to solve the MPC problem
        """
        index = self.lintraj.GetTimeIndex(t)
        self._initialize_mpc(index, x0)
        for n in range(index, index+self.horizon):
            self._add_decision_variables(n)
            self._add_costs(n)
            self._add_constraints(n)

    def _ininitalize_mpc(self, index, x0):
        """
        Initialize the MPC for the current state
        """
        self.prog = MathematicalProgram()
        # Create the initial state and constrain it
        self._dx = [self.prog.NewContinuousVariables(rows = self.state_dim, cols=1, name='state')]
        state0 = self.lintraj.GetState(index)
        dx0 = state0 - x0
        self.prog.AddLinearEqualityConstraint(Aeq = np.eye(self.state_dim), beq=dx0, vars=self._dx[0])
        self.prog.SetInitialGuess(self._dx[0], x0)
        # Clear the remaining variables
        self._du = []
        self._dl = []
        self._ds = []
        self._djl = []

    def _add_decision_variables(self, index):
        """
        Add and initialize decision variables to the program for the current time index
        """
        # Add new variables to the program
        self._dx.append(self.prog.NewContinuousVariables(rows = self.state_dim, cols=1, name='state'))
        self._du.append(self.prog.NewContinuousVariables(rows = self.control_dim, cols=1, name='control'))
        self._dl.append(self.prog.NewContinuousVariables(rows = self.force_dim, cols=1, name='force'))
        self._ds.append(self.prog.NewContinuousVariables(rows = self.slack_dim, cols=1, name='slack'))
        # Add and initialize joint limits
        if self.jlimit_dim > 0:
            self._djl.append(self.prog.NewContinuousVariables(rows = self.jlimit_dim, cols=1, name='joint_limits'))
            self.prog.SetInitialGuess(self._djl[-1], self.lintraj.GetJointLimits(index+1))
        # Initalize the decision variables
        self.prog.SetInitialGuess(self._dl[-1], self.lintraj.GetForces(index+1))
        self.prog.SetInitialGuess(self._ds[-1], self.lintraj.GetSlacks(index+1))
        # Initialize the states and controls (random, nonzero values)
        self.prog.SetInitialGuess(self._dx[-1], np.default_rng().standard_normal(self.state_dim))
        self.prog.SetInitialGuess(self._du[-1], np.default_rng().standard_normal(self.control_dim))

    def _add_constraints(self, index):
        """
        Add the linear contact-implicit dynamics constraints at the specified index
        """
        # Some necessary parameters
        nQ = self.lintraj.plant.multibody.num_positions()
        nC = self.lintraj.plant.num_contacts()
        nF = self.lintraj.plant.num_friction()
        # Add the dynamics and joint limit constraints
        if self.jlimit_dim > 0:
            dl_all = np.concatenate([self._dl[-1], self._djl[-1]], axis=0)
            self.lintraj.getDynamicsConsraint(index).addToProgram(self.prog, self._dx[-2], self._dx[-1], self._du[-1], dl_all)
            self.lintraj.getJointLimitConstraint(index+1).addToProgram(self.prog, self._dx[-1][:nQ], self._djl[-1])
        else:
            self.lintraj.getDynamicsConstraint(index).addToProgram(self.prog, self._dx[-2], self._dx[-1], self._du[-1], self._dl[-1])
        # Update the complementarity cost weights
        self.lintraj.getDistanceConstraint(index+1).cost_weight = self._complementarity_weight
        self.lintraj.getDissipationConstraint(index+1).cost_weight = self._complementarity_weight
        self.lintraj.getFrictionConeConstraint(index+1).cost_weight = self._complementarity_weight
        # Add the contact constraints
        self.lintraj.getDistanceConstraint(index+1).addToProgram(self.prog, self._dx[-1], self._dl[-1][:nC])
        self.lintraj.getDissipationConstraint(index+1).addToProgram(self.prog, np.concatenate([self._dx[-1], self._ds[-1]], axis=0), self._dl[-1][nC:nC+nF])
        self.lintraj.getFrictionConeConstraint(index+1).addToProgram(self.prog, np.concatenate([self._dx[-1], self._dl[-1][:nC+nF]], axis=0), self._ds[-1])

    def _add_costs(self, index):
        """Add cost terms on the most recent variables"""
        # We add quadratic error costs on all variables, regularizing states and controls to 0 and complementarity variables to their trajectory values
        self.prog.AddQuadraticErrorCost(self._state_weight, np.zeros((self.state_dim, )), vars=self._dx[-1], description = 'state_cost')
        self.prog.AddQuadraticErrorCost(self._control_weight, np.zeros((self.control_dim, )), vars=self._du[-1], description = 'control_cost')
        self.prog.AddQuadraticErrorCost(self._force_weight, self.lintraj.GetForces(index+1), vars=self._dl[-1], description = 'force_cost')
        self.prog.AddQuadraticErrorCost(self._slack_weight, self.lintraj.GetSlacks(index+1), vars=self._ds[-1], description = 'slack_cost')
        # Add a cost for joint limits
        if self.jlimit_dim > 0:
            self.prog.AddQuadraticErrorCost(self._jlimit_weight, self.lintraj.GetJointLimits(index+1), vars=self._djl[-1], description = 'joint_limit_cost')

    def solve(self):
        """Solves the created MPC problem"""
        #TODO: Allow user to set options for solving the problem
        return Solve(self.prog)

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

if __name__ == "__main__":
    print("Hello from MPC!")