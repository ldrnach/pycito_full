"""
Class for Linear Contact-Implicit MPC

January 26, 2022
Luke Drnach
"""
#TODO: Allow user to specify MPC solver, settings

import numpy as np

from pydrake.all import MathematicalProgram, Solve

import pycito.utilities as utils
import pycito.trajopt.constraints as cstr
import pycito.controller.mlcp as mlcp

class LinearizedContactTrajectory():
    def __init__(self, plant, xtraj, utraj, ftraj, jltraj=None, lcp=mlcp.CostRelaxedMixedLinearComplementarity):
        self.plant = plant
        self.lcp = lcp
        # Store the trajectory values
        self.time, self.state = utils.GetKnotsFromTrajectory(xtraj)
        self.control = utraj.vector_values(self.time)
        forces = ftraj.vector_values(self.time)
        self.forces = forces[:self.plant.num_contacts() + self.plant.num_friction(), :]
        self.slacks = forces[self.plant.num_contacts() + self.plant.num_friction():, :]
        if jltraj is not None:
            self.jointlimits = jltraj.vector_values(self.time)
            self._linearize_joint_limits()
        else:
            self.jointlimits = None
            self.joint_limit_cstr = None
        # Store the linearized parameter values
        self._linearize_dynamics()
        self._linearize_normal_distance()
        self._linearize_maximum_dissipation()
        self._linearize_friction_cone()

    def _linearize_dynamics(self):
        """Store the linearization of the dynamics constraints"""
        self.dynamics_cstr = []
        force_idx = 2*self.state.shape[0] + self.control.shape[0] + 1
        dynamics = cstr.BackwardEulerDynamicsConstraint(self.plant)
        if self.jointlimits is not None:
            force = np.concatenate([self.forces, self.jointlimits], axis=0)
        else:
            force = self.forces
        for n in range(self.time.size-1):
            h = self.time[n+1] - self.time[n]
            A, _ = dynamics.linearize(h, self.state[:,n], self.state[:, n+1], self.control[:,n+1], force[:, n+1])
            b = A[:,0] - A[:, force_idx:].dot(force[:, n+1])
            A = A[:, 1:]
            self.dynamics_cstr.append(cstr.LinearImplicitDynamics(A, b))

    def _linearize_normal_distance(self):
        """Store the linearizations for the normal distance constraint"""
        distance = cstr.NormalDistanceConstraint(self.plant)
        self.distance_cstr = []
        B = np.zeros((self.plant.num_contacts(), self.plant.num_contacts()))
        for x in self.state.transpose():
            A, c = distance.linearize(x)
            self.distance_cstr.append(self.lcp(A, B, c))

    def _linearize_maximum_dissipation(self):
        """Store the linearizations for the maximum dissipation function"""
        dissipation = cstr.MaximumDissipationConstraint(self.plant)
        self.dissipation_cstr = []
        B = np.zeros((self.plant.num_friction(), self.plant.num_friction()))
        for x, s in zip(self.state.transpose(), self.slacks.transpose()):
            A, c = dissipation.linearize(x, s)
            c -= A[:, x.shape[0]:].dot(s)       #Correction term for LCP 
            self.dissipation_cstr.append(self.lcp(A, B, c))

    def _linearize_friction_cone(self):
        """Store the linearizations for the friction cone constraint function"""
        friccone = cstr.FrictionConeConstraint(self.plant)
        self.friccone_cstr = []
        B = np.zeros((self.plant.num_contacts(), self.plant.num_contacts()))
        for x, f in zip(self.state.transpose(), self.forces.transpose()):
            A, c = friccone.linearize(x, f)
            c -= A[:, x.shape[0]:].dot(f)   #Correction term for LCP
            self.friccone_cstr.append(self.lcp(A, B, c))

    def _linearize_joint_limits(self):
        """Store the linearizations of the joint limits constraint function"""
        jointlimits = cstr.JointLimitConstraint(self.plant)
        self.joint_limit_cstr = []
        B = np.zeros((jointlimits.num_joint_limits, jointlimits.num_joint_limits))
        nQ = self.plant.multibody.num_positions()
        for x in self.state.transpose():
            A, c = jointlimits.linearize(x[:nQ])
            self.joint_limit_cstr.append(self.lcp(A, B, c))
    
    def getTimeIndex(self, t):
        """Return the index of the last timepoint less than the current time"""
        return np.argmax(self.time > t) - 1

    def getDynamicsConstraint(self, index):
        """Returns the linear dynamics constraint at the specified index"""
        index = min(index, len(self.dynamics_cstr))
        return self.dynamics_cstr[index]

    def getDistanceConstraint(self, index):
        """Returns the normal distance constraint at the specified index"""
        index = min(index, len(self.distance_cstr))
        return self.distance_cstr[index]

    def getDissipationConstraint(self, index):
        """Returns the maximum dissipation constraint at the specified index"""
        index = min(index, len(self.dissipation_cstr))
        return self.dissipation_cstr[index]

    def getFrictionConeConstraint(self, index):
        """Returns the friction cone constraint at the specified index"""
        index = min(index, len(self.friccone_cstr))
        return self.friccone_cstr[index]

    def getJointLimitConstraint(self, index):
        """Returns the joint limit constraint at the specified index"""
        index = min(index, len(self.joint_limit_cstr))
        return self.joint_limit_cstr[index]

    def getState(self, index):
        """
        Return the state at the given index
        """
        index = min(index, self.state.shape[1])
        return self.state[:, index]

    def getControl(self, index):
        """
        Return the control at the given index
        """
        index = min(index, self.control.shape[1])
        return self.control[:, index]

    def getForce(self, index):
        """
        Return the force at the given index
        """
        index = min(index, self.forces.shape[1])
        return self.forces[:, index]

    def getSlack(self, index):
        """
        Return the velocity slacks at the given index
        """
        index = min(index, self.slack.shape[1])
        return self.slacks[:, index]

    @property
    def state_dim(self):
        return self.state.shape[0]
    
    @property
    def control_dim(self):
        return self.control.shape[0]

    @property
    def force_dim(self):
        return self.forces.shape[0]

    @property
    def slack_dim(self):
        return self.slacks.shape[0]

    @property
    def jlimit_dim(self):
        if self.jointlimits is not None:
            self.jointlimits.shape[0]
        else:
            return 0

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