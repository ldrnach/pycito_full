"""
Contact model estimation

Luke Drnach
February 14, 2022
"""
#TODO: Refactor with MPC.ReferenceTrajectory
#TODO: Integrate constraints with ContactEstimationTrajectory
#TODO: Finish ContactModelEstimator by integrating with ContactEstimationTrajectory
import numpy as np

from pydrake.all import MathematicalProgram

import pycito.trajopt.constraints as cstr
import pycito.trajopt.complementarity as cp

class ObservationTrajectory():
    """
    Container for easily storing trajectories for contact estimation
    """
    def __init__(self):
        """Initialize the observation trajectory"""
        self._time = []
        self._state = []
        self._control = []

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
        return self._state[index]

    def getControl(self, index):
        """
        Return the control at the given index, returning the last control if the index is out of bounds
        """
        index = min(max(0,index), self.num_timesteps-1)
        return self._control[index]

    def add_sample(self, time, state, control):
        """
        Add a new sample to the trajectory. 
        Requires:
            time is greater than the previous timestep (the overall time is monotonically increasing)
            state has the same dimension as the previous state
            control has the same dimension as the previous control
        """
        if self._time is None:
            self._time = [time]
            self._state = [state]
            self._control = [control]
        else:
            assert time > self._time[-1], "time must be monotonically increasing"
            assert state.shape[0] == self._state.shape[0], f"state must have shape ({self._state.shape[0]}, )"
            assert control.shape[0] == self._control.shape[0], f"control must have shape ({self._control.shape[0]}, )"
            self._time.append(time)
            self._state.append(state)
            self._control.append(control)

    @property
    def num_timesteps(self):
        return len(self._time)

    @property
    def state_dim(self):
        if self._state is not []:
            return self._state[0].shape[0]
        else:
            return None

    @property
    def control_dim(self):
        if self._control is not []:
            return self._control[0].shape[0]
        else:
            return None

class ContactEstimationTrajectory(ObservationTrajectory):
    def __init__(self, plant):
        super(ContactEstimationTrajectory, self).__init__()
        self._plant = plant
        self._context = plant.multibody.CreateDefaultContext()
        # Setup the constraint parameter lists
        self._dynamics_cstr = []
        self._distance_cstr = []
        self._dissipation_cstr = []
        self._friction_cstr = []

    def add_sample(self, time, state, control):
        """
        Append a new point to the trajectory
        Also appends new parameters for the dynamics, normal distance, dissipation, and friction coefficient
        """
        super(ContactEstimationTrajectory, self).add_sample(time, state, control)
        self._add_dynamics()
        self._add_distance()
        self._add_dissipation()
        self._add_friction()

    def _add_dynamics(self):
        """
        Add a set of linear system parameters to evaluate the dynamics defect 
        """
        state = self._state[-1]
        dt = self._time[-1] - self._time[-2]
        control = self._control[-1]
        # Get the Jacobian matrix - the "A" parameter
        self._plant.multibody.SetPositionsAndVelocities(self._context, state)
        Jn, Jt = self._plant.GetContactJacobians(self._context)
        A = np.concatenate([Jn.T, Jt.T], axis=0)
        # Now get the dynamics defect
        if self._plant.has_joint_limits:
            forces = np.zeros((A.shape[1] + self._plant.joint_limit_jacobian().shape[1], ))
        else:
            forces = np.zeros((A.shape[1], ))
        b = cstr.BackwardEulerDynamicsConstraint.eval(self._plant, self._context, dt, self._state[-2], state, control, forces)
        # Append - wrap this in it's own constraint
        self._dynamics_cstr.append((A, b))

    def _add_distance(self):
        """
        Add a set of linear system parameters to evaluate the normal distance defect
        """
        # Get the distance vector
        self._plant.multibody.SetPositionsAndVelocities(self._context, self._state[-1])
        b = self._plant.GetNormalDistances(self._context)
        self._distance_cstr.append(b)

    def _add_dissipation(self):
        """
        Add a set of linear system parameters to evaluate the dissipation defect
        """
        # Get the tangent jacobian
        self._plant.multibody.SetPositionsAndVelocities(self._context, self._state[-1])
        _, Jt = self.plant.GetContactJacobians(self._context)
        dq = self._state[-1][self._plant.multibody.num_positions():]
        b = Jt.dot(dq)
        self._dissipation_cstr.append(b)

    def _add_friction(self):
        """
        Add a set of linear system parameters to evaluate the friction cone defect
        """
        # Append the friction coefficients
        self._plant.multibody.SetPositionsAndVelocities(self._context, self._state[-1])
        mu = self.plant.GetFrictionCoefficients(self._context)
        self._friction_cstr.append(np.diag(mu))

    def getDynamicsConstraint(self, index):
        """
        Returns the dynamics constraint at the specified index
        """
        assert index >= 0 and index < len(self._dynamics_cstr), "index out of bounds"
        return self._dynamics_cstr[index]

    def getDistanceConstraint(self, index):
        """
        Return the distance constraint at the specified index
        """
        assert index >= 0 and index < len(self._distance_cstr), "index out of bounds"
        return self._distance_cstr[index]

    def getDissipationConstraint(self, index):
        """
        Return the dissipation constraint at the specified index
        """
        assert index >= 0 and index < len(self._dissipation_cstr), "index out of bounds"
        return self._dissipation_cstr[index]

    def getFrictionConstraint(self, index):
        """
        Return the friction cone constraint at the specified index
        """
        assert index >= 0  and index < len(self._friction_cstr), "index out of bounds"
        return self._friction_cstr[index]

class ContactModelEstimator():
    def __init__(self, esttraj, horizon):
        """
        Arguments:
            esttraj: a ContactEstimationTrajectory object
            horizon: a scalar indicating the reverse horizon to use for estimation
        """
        self.traj = esttraj
        self.horizon = horizon
        # Internal variables - program and decision variables
        self._prog = None
        self._distance_weights = None
        self._friction_weights = None
        self._normal_forces = None
        self._friction_forces = None
        self._velocity_slacks = None
        self._relaxation_vars = None
        # Cost weights
        self._relax_cost = 1.
        self._force_cost = 1.

    def create_estimator(self):
        """
        Create a mathematical program to solve the estimation problem
        """
        self._prog = MathematicalProgram()

    @property
    def relaxedcost(self):
        return self._relax_cost

    @relaxedcost.setter
    def relaxedcost(self, val):
        assert isinstance(val, [int, float]) and val >=0, f"val must be a nonnegative float or a nonnegative int"
        self._relax_cost = val

    @property
    def forcecost(self):
        return self._force_cost

    @forcecost.setter
    def forcecost(self, val):
        assert isinstance(val, [int, float]) and val >= 0, "val must be a nonnegative float or nonnegative int"
        self._force_cost = val

    @property
    def forces(self):
        if self._normal_forces is not None:
            return np.concatenate([self._normal_forces, self._friction_forces], axis=0)
        else:
            return None

    @property
    def velocities(self):
        if self._velocity_slacks is not None:
            return self._velocity_slacks
        else:
            return None

class SemiparametricFrictionConeConstraint():
    def __init__(self, friccoeff, kernel, duplicator, ncptype=cp.NonlinearVariableSlackComplementarity):
        assert friccoeff.shape[0] == kernel.shape[0], 'Friction coefficient and kernel matrix must have the same number of rows'
        assert duplicator.shape[0] == friccoeff.shape[0], 'The duplication matrix must have as many rows as there are friction coefficients'
        self.friccoeff = friccoeff
        self.kernel = kernel            # The kernel matrix
        self.duplicator = duplicator    # Friction force duplication/coupling matrix
        self.ncptype = ncptype              # Nonlinear complementarity constraint implementation
        self._ncp_cstr = None

    def addToProgram(self, prog, *args):
        dvars = np.concatenate(args)
        xvars, zvars = np.split(dvars, [self.num_weights + self.num_normals + self.num_friction])
        self._ncp_cstr = self.ncptype(self.eval_frictioncone, xdim = xvars.shape[0], zdim = zvars.shape[0])
        self._ncp_cstr.set_description('SemiparametricFrictionCone')
        # Add the nonlinear complementarity constraint to the program
        self._ncp_cstr.addToProgram(prog, xvars, zvars)
        # Add the nonnegativity constraint to the program - linear constraint
        weights = xvars[:self.num_weights]
        prog.AddLinearConstraint(A = self.kernel, lb = -self.friccoeff, ub = np.full(self.friccoeff.shape, np.inf), vars=weights).evaluator().set_description('friction_coeff_nonnegativity')

    def eval_frictioncone(self, dvals):
        # Separate out the components of the constraint
        weights, normal_force, friction_force = np.split(dvals, np.cumsum([self.num_weights, self.num_normals]))
        # Calculate the "updated" friction cone
        mu = np.diag(self.eval_friction_coeff(weights))
        # Friction cone error 
        return mu.dot(normal_force) - self.duplicator.dot(friction_force)

    def eval_friction_coeff(self, weights):
        """Return the semiparametric friction coefficient"""
        return self.friccoeff + self.kernel.dot(weights)

    @property
    def num_normals(self):
        return self.friccoeff.size

    @property
    def num_weights(self):
        return self.kernel.shape[1]

    @property
    def num_friction(self):
        return self.duplicator.shape[1]

    @property
    def relax(self):
        if self._ncp_cstr is not None:
            return self._ncp_cstr.var_slack
        else:
            return None

    @property
    def cost_weight(self):
        if self._ncp_cstr is not None:
            return self._ncp_cstr.cost_weight
        else:
            return None

    @cost_weight.setter
    def cost_weight(self, val):
        if self._ncp_cstr is not None:
            self._ncp_cstr.cost_weight = val

if __name__ == '__main__':
    print("Hello from contactestimator!")