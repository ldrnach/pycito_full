"""
Contact model estimation

Luke Drnach
February 14, 2022
"""
#TODO: Update EstimatedContactModelRectifier
#TODO: Refactor ContactModelEstimator to reuse and update the program



import numpy as np
import copy, warnings
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, SnoptSolver
from pydrake.all import PiecewisePolynomial as pp

import pycito.trajopt.constraints as cstr
import pycito.trajopt.complementarity as cp
import pycito.controller.mlcp as mlcp
from pycito.systems.contactmodel import SemiparametricContactModel
from pycito.controller.optimization import OptimizationMixin
import pycito.decorators as deco
import pycito.utilities as utils

class SemiparametricFrictionConeConstraint():
    def __init__(self, friccoeff, kernel, duplicator):
        assert friccoeff.shape[0] == kernel.shape[0], 'Friction coefficient and kernel matrix must have the same number of rows'
        assert duplicator.shape[0] == friccoeff.shape[0], 'The duplication matrix must have as many rows as there are friction coefficients'
        self.friccoeff = friccoeff
        self.kernel = kernel            # The kernel matrix
        self.duplicator = duplicator    # Friction force duplication/coupling matrix
        self._ncp_cstr = None

    def addToProgram(self, prog, xvars, zvars, rvars=None):
        self._ncp_cstr = cp.NonlinearVariableSlackComplementarity(self.eval_frictioncone, xdim = xvars.shape[0], zdim = zvars.shape[0])
        self._ncp_cstr.set_description('SemiparametricFrictionCone')
        # Add the nonlinear complementarity constraint to the program
        self._ncp_cstr.addToProgram(prog, xvars, zvars, rvars=rvars)
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

class ContactTrajectory():
    """
    Container for easily storing trajectories for contact estimation
    """
    def __init__(self):
        """Initialize the observation trajectory"""
        self._time = []
        self._contactpoints = []
        self._forces = []
        self._slacks = []
        self._feasibility = []
        self._distance_error = []
        self._friction_error = []

    def save(self, filename):
        """
        Save the ContactTrajectory to a file
        """
        utils.save(filename, self)

    @staticmethod
    def load(filename):
        """
        Load the ContactTrajectory from a file
        """
        data = utils.load(filename)
        assert isinstance(data, ContactTrajectory), 'file does not contain a ContactTrajectory'
        return data

    def subset(self, start, stop, *args):
        """Return a subset of the original contact trajectory"""
        new = self.__class__(*args)
        new._time = self._time[start:stop]
        new._contactpoints = self._contactpoints[start:stop]
        new._forces = self._forces[start:stop]
        new._slacks = self._slacks[start:stop]
        new._feasibility = self._feasibility[start:stop]
        return new

    def set_at(self, lst, idx, x, default=None):
        """
        Set the value of a list at the given index, extending the list if necessary
        
        Arguments:
            lst: the list 
            idx: (int) the desired index. If idx is a float, it is interpreted as the 'timestamp' and set_at calculates the nearest timepoint and sets the value to the corresponding index
            x: the value to set
            default (optional): the default value used to fill in unset values in the array. default = None
        """
        idx = self.getTimeIndex(idx) if isinstance(idx, float) else idx
        if len(lst) <= idx:
            lst.extend([default] * (idx - len(lst) + 1))
        lst[idx] = x

    def get_at(self, lst, start_idx, stop_idx=None):
        stop_idx = start_idx+1 if stop_idx is None else stop_idx
        return lst[start_idx:stop_idx]

    def add_contact_sample(self, time, contacts):
        """
        Add a contact sample and the corresponding timestep
        add_contact_sample updates the time axis and inserts a contact sample at the corresponding index

        Arguments:
            time: (float) the timestamp at which the contacts are measured
            contacts: (array) the contact points at the current time
        """
        self.add_time(time)
        self.set_contacts(self.num_timesteps-1, contacts)

    def add_time(self, time):
        """
        Add a new time stamp to the time array
        
        Arguments:
            time: (1,) array or float, the timestamp to append

        Requirements:
            time must be a float or a scalar array, and time must be greater than the current value at the end of self._time - i.e. the overall sequence of timesteps must be monotonically increasing
        """
        time = np.asarray(time)
        assert time.size == 1, f'timestamp must be a scalar'
        if self._time == []:
            self._time = [time]
        else:
            assert time > self._time[-1], f'Time must be monotonically increasing'
            self._time.append(time)

    def set_contacts(self, index, contacts):
        """
        Update contacts and the specified index
        
        Arguments:
            index: (int) the index at which to set the value of the contacts
            contacts: the values to set
        """
        self.set_at(self._contactpoints, index, contacts)

    def set_force(self, index, force):
        """
        set contact forces and the specified index
        
        Arguments:
            index: (int) the index at which to set the value 
            force: the values to set
        """
        self.set_at(self._forces, index, force)

    def set_dissipation(self, index, dslack):
        """
        set dissipation slacks and the specified index
        
        Arguments:
            index: (int) the index at which to set the value 
            dslack: the values to set
        """
        self.set_at(self._slacks, index, dslack)

    def set_feasibility(self, index, rslack):
        """
        set feasibility values and the specified index
        
        Arguments:
            index: (int) the index at which to set the value 
            rslack: the feasibility value to set
        """
        self.set_at(self._feasibility, index, rslack)

    def getTimeIndex(self, t):
        """
        Return the index of the last timepoint less than the current time
        
        Argument:
            t (float): the test timepoint

        Return value:
            idx (int) the index of the last timepoint less than t
        """
        if t < self._time[0]:
            return 0
        elif t == self._time[-1]:
            return self.num_timesteps - 1
        elif t > self._time[-1]:
            return self.num_timesteps
        else:
            return np.argmax(np.array(self._time) > t) - 1

    def get_contacts(self, start_idx, stop_idx = None):
        """
        Return a list of contact points at the specified index

        Arguments:
            start_idx (int): the first index to return
            stop_idx (int, optional): the last index to return (by default, start_index)
        
        Returns:
            lst: contact points between [start_idx, stop_idx)
        """
        return self.get_at(self._contactpoints, start_idx, stop_idx)

    def get_forces(self, start_idx, stop_idx = None):
        """
        Return a list of contact forces at the specified index

        Arguments:
            start_idx (int): the first index to return
            stop_idx (int, optional): the last index to return (by default, start_index)
        
        Returns:
            lst: contact forces between [start_idx, stop_idx)
        """
        return self.get_at(self._forces, start_idx, stop_idx)

    def get_feasibility(self, start_idx, stop_idx=None):
        """
        Return a list of feasibility values at the specified index

        Arguments:
            start_idx (int): the first index to return
            stop_idx (int, optional): the last index to return (by default, start_index)
        
        Returns:
            lst: feasibility values between [start_idx, stop_idx)
        """
        return self.get_at(self._feasibility, start_idx, stop_idx)

    def get_dissipation(self, start_idx, stop_idx=None):
        """
        Return a list of velocity dissipation slacks at the specified index

        Arguments:
            start_idx (int): the first index to return
            stop_idx (int, optional): the last index to return (by default, start_index)
        
        Returns:
            lst: velocity dissipation slacks between [start_idx, stop_idx)
        """
        return self.get_at(self._slacks, start_idx, stop_idx)

    def get_distance_error(self, start_idx, stop_idx=None):
        """
        Return a list of signed distance errors at the specified index

        Arguments:
            start_idx (int): the first index to return
            stop_idx (int, optional): the last index to return (by default, start_index)
        
        Returns:
            lst: signed distance errors between [start_idx, stop_idx)
        """
        return self.get_at(self._distance_error, start_idx, stop_idx)

    def get_friction_error(self, start_idx, stop_idx=None):
        """
        Return a list of friction coefficient errors at the specified index

        Arguments:
            start_idx (int): the first index to return
            stop_idx (int, optional): the last index to return (by default, start_index)
        
        Returns:
            lst: friction coefficient errors between [start_idx, stop_idx)
        """
        return self.get_at(self._friction_error, start_idx, stop_idx)

    def set_distance_error(self, index, derr):
        """
        set distance error values and the specified index
        
        Arguments:
            index: (int) the index at which to set the value 
            derr: the distance error value to set
        """
        self.set_at(self._distance_error, index, derr)

    def set_friction_error(self, index, ferr):
        """
        set friction error values and the specified index
        
        Arguments:
            index: (int) the index at which to set the value 
            ferr: the distance error value to set
        """
        self.set_at(self._friction_error, index, ferr)
    
    @property
    def num_timesteps(self):
        return len(self._time)

class ContactEstimationTrajectory(ContactTrajectory):
    def __init__(self, plant, initial_state):
        super(ContactEstimationTrajectory, self).__init__()
        self._FTOL = 1e-8
        self._plant = plant
        self._context = plant.multibody.CreateDefaultContext()
        # Store the duplication matrix
        self._D = self._plant.duplicator_matrix()
        # Store the terrain kernel matrices
        self.contact_model = plant.terrain
        assert isinstance(self.contact_model, SemiparametricContactModel), f"plant does not contain a semiparametric contact model"
        # Setup the constraint parameter lists
        self._dynamics_cstr = []
        self._distance_cstr = []
        self._dissipation_cstr = []
        self._friction_cstr = []

        # Store the last state for calculating dynamics - assume it was static before we started moving
        self._last_state = initial_state
        self._plant.multibody.SetPositionsAndVelocities(self._context, initial_state)
        cpt = self._plant.get_contact_points(self._context)
        self.add_contact_sample(0., np.hstack(cpt))    # Add the contact point
        u, fN = plant.static_controller(qref = initial_state[:plant.multibody.num_positions()])
        N = plant.multibody.CalcGravityGeneralizedForces(self._context)
        B = plant.multibody.MakeActuationMatrix()
        Jn, Jt = plant.GetContactJacobians(self._context)
        A = np.concatenate((Jn.T, Jt.T), axis=1)
        self._dynamics_cstr = [(A, -N - B.dot(u))]
        # Set the initial guesses for force and feasibility
        f_guess = np.zeros((self.num_contacts + self.num_friction, ))
        f_guess[:self.num_contacts] = fN
        self.set_force(0, f_guess)
        err = A.dot(f_guess) + (N + B.dot(u))
        self.set_feasibility(0, np.ones(1,)*err.max())
        # Add the contact constraints
        self._append_distance()
        self._append_dissipation()
        self._append_friction()

    def save(self, filename):
        # Save the estimation trajectory to the disk
        var_dict = vars(self)
        plant_copy = copy.deepcopy(self._plant)
        var_dict['_plant'] = type(self._plant).__name__
        var_dict.pop('_context')
        var_dict['isContactEstimationTrajectory'] = True
        utils.save(filename, var_dict)
        # Put the plant back in
        self._plant = plant_copy
        self._context = self._plant.multibody.CreateDefaultContext()

    @classmethod
    def load(cls, plant, filename):
        data = utils.load(utils.FindResource(filename))
        # Type checking
        if 'isContactEstimationTrajectory' not in data: 
            raise ValueError(f"{filename} does not contain a ContactEstimationTrajectory")       
        if not data['isContactEstimationTrajectory']:
            raise ValueError(f"{filename} does not contain a ContactEstimationTrajectory")
        if data['_plant'] != type(plant).__name__:
            raise ValueError(f"{filename} was made with a {data['_plant']} plant model, but a {type(plant).__name__} was given")
        # Create the new contact estimation trajectory
        newinstance = cls(plant, data['_last_state'])
        data.pop('isContactEstimationTrajectory')
        data.pop('_plant')
        for key, value in data.items():
            setattr(newinstance, key, value)
        return newinstance

    def saveContactTrajectory(self, filename):
        """
        Saves only a copy of the ContactTrajectory superclass
        """
        traj = ContactTrajectory()
        traj._time = self._time
        traj._contactpoints = self._contactpoints
        traj._forces = self._forces
        traj._slacks = self._slacks
        traj._feasibility = self._feasibility
        traj._distance_error = self._distance_error
        traj._friction_error = self._friction_error
        traj.save(filename)

    def subset(self, start, stop):
        """Get a subset of the estimation trajectory"""
        new = super(ContactEstimationTrajectory, self).subset(start, stop, self._plant, self._last_state)
        # Slice the model parameters
        new._dynamics_cstr = self._dynamics_cstr[start:stop]
        new._distance_cstr = self._distance_cstr[start:stop]
        new._dissipation_cstr = self._dissipation_cstr[start:stop]
        new._friction_cstr = self._friction_cstr[start:stop]
        new._distance_error = self._distance_error[start:stop]
        new._friction_error = self._friction_error[start:stop]
        return new

    def append_sample(self, time, state, control):
        """
        Append a new point to the trajectory
        Also appends new parameters for the dynamics, normal distance, dissipation, and friction coefficient
        """
        # Reset the context, and add the contact points
        self._plant.multibody.SetPositionsAndVelocities(self._context, state)
        cpts = self._plant.get_contact_points(self._context)
        self.add_contact_sample(time, np.hstack(cpts))
        # Add the dynamics constraint last, NOTE that adding the dynamics manipulates the context
        self._append_dynamics(state, control)
        self._plant.multibody.SetPositionsAndVelocities(self._context, state)
        # Also add the distance, dissipation, and friction constraints
        self._append_distance()
        self._append_dissipation()
        self._append_friction()

    def _append_dynamics(self, state, control):
        """
        Add a set of linear system parameters to evaluate the dynamics defect 
        """
        dt = self._time[-1] - self._time[-2]
        # Get the Jacobian matrix - the "A" parameter
        self._plant.multibody.SetPositionsAndVelocities(self._context, state)
        Jn, Jt = self._plant.GetContactJacobians(self._context)
        A = dt*np.concatenate([Jn.T, Jt.T], axis=1)
        # Now get the dynamics defect
        if self._plant.has_joint_limits:
            forces = np.zeros((A.shape[1] + self._plant.joint_limit_jacobian().shape[1], ))
        else:
            forces = np.zeros((A.shape[1], ))
        b = cstr.BackwardEulerDynamicsConstraint.eval(self._plant, self._context, dt, self._last_state, state, control, forces)
        # Take only the velocity components
        b = b[self._plant.multibody.num_positions():]
        # Append - wrap this in it's own constraint
        self._dynamics_cstr.append((A, b))
        # Save the state for the the next call to append_dynamics
        self._last_state = state
        # Add a guess to the reaction forces
        f = np.linalg.lstsq(A, b, rcond=None)[0]
        # Reorganize the friction forces
        fN, fT = np.split(f, [self.num_contacts])
        FD = self._plant.friction_discretization_matrix()
        fT = FD.T.dot(FD.dot(fT))
        fT[fT < self._FTOL] = 0
        f = np.concatenate([fN, fT], axis=0)
        self._forces.append(f)
        # Add the feasibility guess
        err = A.dot(f) - b
        self._feasibility.append(np.ones(1,)*err.max())

    def _append_distance(self):
        """
        Add a set of linear system parameters to evaluate the normal distance defect
        """
        # Get the distance vector
        b = self._plant.GetNormalDistances(self._context)
        self._distance_cstr.append(b)
        # Add an element to the distance error 
        d_err = np.zeros_like(b)
        d_err[b < 0.] = -b[b < 0.]
        self._distance_error.append(d_err)
        # Increase the feasibility, if necessary (guarantee that all constraints are trivially satisfied at the initial point)
        fN = self._forces[-1][:self.num_contacts]
        w = b * fN
        self._feasibility[-1] += np.max(np.abs(w))

    def _append_dissipation(self):
        """
        Add a set of linear system parameters to evaluate the dissipation defect
        """
        # Get the tangent jacobian
        _, Jt = self._plant.GetContactJacobians(self._context)
        v = self._plant.multibody.GetVelocities(self._context)
        b = Jt.dot(v)
        self._dissipation_cstr.append(b)
        # Append to the dissipation slacks - TODO: Update how the slack is determined for multiple contacts
        vs = np.max(-b) * np.ones((self._D.shape[0],))
        self._slacks.append(vs)
        # Increase the feasibility to ensure the constraint is trivially satisfied
        fT = self._forces[-1][self.num_contacts:]
        w = (self._D.T.dot(vs) + b) * fT
        self._feasibility[-1] += np.max(np.abs(w))

    def _append_friction(self):
        """
        Add a set of linear system parameters to evaluate the friction cone defect
        """
        # Append the friction coefficients
        mu = self._plant.GetFrictionCoefficients(self._context)
        self._friction_cstr.append(mu)
        # Use the most recent value of the forces to calculate the friction cone defecit
        fN, fT = np.split(self._forces[-1], [self.num_contacts])
        fc = np.diag(mu).dot(fN) - self._D.dot(fT)
        mu_err = np.zeros_like(mu)
        err_index = fc < 0 and fN > 0
        if np.any(err_index):
            mu_err[fc < 0 and fN > 0] = -fc[fc < 0 and fN > 0]/fN[fc < 0 and fN > 0]
        self._friction_error.append(mu_err)
        # Update the feasibility to ensure the nonlinear constraint is trivially satisfied
        g = self._slacks[-1]
        w = (mu * fN - self._D.dot(fT)) * g
        self._feasibility[-1] += np.max(np.abs(w))

    def getDynamicsConstraint(self, index):
        """
        Returns the dynamics constraint at the specified index
        The dynamics constraint takes the form
            A*x  = b
        where A is the contact Jacobian, x is the reaction forces, and b are the net external reaction force

        Arguments:
            index (int): a scalar

        Return Value:
            (A,b) - a tuple of dynamics constraint parameters
            A: (nV, nC + nF) array, the scaled contact Jacobian, where nV is the number of velocity variables, nC is the number of contacts, and nF is the number of friction components
            b: (nC + nF) array, the scaled net generalized reaction force
        """
        assert index >= 0 and index < len(self._dynamics_cstr), "index out of bounds"
        return self._dynamics_cstr[index]

    def getDistanceConstraint(self, index):
        """
        Return the distance constraint at the specified index

        Arguments:
            index (int): the scalar index at which to get the normal distance

        Return value:
            d: (nC,) array, the contact distances for each of the nC contact points
        """
        assert index >= 0 and index < len(self._distance_cstr), "index out of bounds"
        return self._distance_cstr[index]

    def getDissipationConstraint(self, index):
        """
        Return the dissipation constraint at the specified index

        Argument:
            index (int): the scalar index at which to get the dissipation constraint parameters

        Return values:
            (D, g) tuple of constraint parameters
                D: (nT, nC) array, duplication matrix duplicating the velocities of the nC contact points along the nT tangential directions
                g: (nC,) array, the maximum tangential velocity slack variables
        """
        assert index >= 0 and index < len(self._dissipation_cstr), "index out of bounds"
        return self._D.transpose(), self._dissipation_cstr[index]

    def getFrictionConstraint(self, index):
        """
        Return the friction cone constraint at the specified index

        Argument:
            index (int): scalar index at which to get the friction coefficient

        Return values:
            (D, m) tuple of constraint parameters
            D: (nC, nT), numpy array, the duplication matrix combining the nT tangential reaction force vectors into nC friction values
            m: nC-list of (1,) numpy arrays, the friction coefficients for each of the nC contact points
        """
        assert index >= 0  and index < len(self._friction_cstr), "index out of bounds"
        return self._D, self._friction_cstr[index]

    def getForceGuess(self, index):
        """
        Return the initial guess for the reaction forces at the specified index

        Arguments:
            index (int): a scalar index

        Return Value:
            f: (nC+nF,) array, a nonnegative guess at the reaction forces at the specified index
        """
        if index < len(self._forces):
            return self._forces[index]
        else:
            A, b = self.getDynamicsConstraint(index)
            f = np.linalg.lstsq(A, b, rcond=None)[0]
            f[f < 0] = 0
            return f

    def getDissipationGuess(self, index):
        """
        Return the initial guess for the dissipation slacks at the specified index

        Arguments:
            index (int): a scalar index variables

        Return value:
            v: (nF, ) array of nonnegative guesses for the tangential velocity slack variable, where nF is the number of friction forces
        """
        if index < len(self._slacks):
            return self._slacks[index]
        else:
            D, v = self.getDissipationConstraint(index)
            return np.max(-v)*np.ones((D.shape[1], ))

    def getFeasibilityGuess(self, index):
        """
        Return an initial guess for the feasibility relaxation variable

        Argument:
            index: (int) a nonnegative scalar

        Return value:
            f: (1,) nonnegative scalar numpy array, the guess for the solution feasibility
        """
        if index < len(self._feasibility):
            return self._feasibility[index]
        else:
            return np.ones((1,))

    def getContactKernels(self, start, stop=None):
        """
        Return the surface and friction kernel matrices between the specified indices

        Arguments:
            start (int): the time index at which to start using sample points to calculate the kernel matrix
            stop (int, optional): the time index at which to stop using sample points to calculate the kernel matrix (default: start + 1)
        
        Return values:
            surface_kernel: (N, N) array, the kernel matrix for the surface kernel
            friction_kernel: (N, N) array, the kernel matrix for the friction kernel
        """
        if self.num_timesteps == 0:
            return np.zeros((0,0)), np.zeros((0,0))
        # Get the contact points
        cpts = np.concatenate(self.get_contacts(start, stop), axis=1)   
        # Calculate the surface and friction kernel matrices
        return self.contact_model.surface_kernel(cpts), self.contact_model.friction_kernel(cpts)
    
    @property
    def num_contacts(self):
        """Returns the number of contact points in the model"""
        return self._plant.num_contacts()

    @property
    def num_friction(self):
        """Returns the total number of friction force components in the model"""
        return self._plant.num_friction()

class ContactModelEstimator(OptimizationMixin):
    def __init__(self, esttraj, horizon, lcp = mlcp.VariableRelaxedPseudoLinearComplementarityConstraint):
        """
        Arguments:
            esttraj: a ContactEstimationTrajectory object
            horizon: a scalar indicating the reverse horizon to use for estimation
        """
        super(ContactModelEstimator, self).__init__()
        self.traj = esttraj
        self.maxhorizon = horizon
        self.lcp = lcp
        self._clear_program()
        # Cost weights
        self._relax_cost_weight = 1.
        self._force_cost_weight = 1.
        self._distance_cost_weight = 1.
        self._friction_cost_weight = 1.
        self._solver = SnoptSolver()

    def estimate_contact(self, t, x, u):
        # Append new samples
        if t <= self.traj._time[-1]:
            return self.get_contact_model_at_time(t)

        self.traj.append_sample(t, x, u)
        # Create contact estimation problem 
        print(f"Creating contact estimation problem")
        self.create_estimator()
        print(f"Solving contact estimation")
        result = self.solve()
        self.update_trajectory(t, result)
        return self.get_updated_contact_model(result)

    def get_contact_model_at_time(self, t):
        """Get the contact model at the specified index"""
        index = self.traj.getTimeIndex(t)
        model = copy.deepcopy(self.traj.contact_model)
        start = max(0, index - self.maxhorizon)
        Kd, Kf = self.traj.getContactKernels(start, index + 1)
        cpts = np.column_stack(self.traj.get_contacts(start, index + 1))
        forces = np.column_stack(self.traj.get_forces(start, index + 1))
        derr = np.row_stack(self.traj.get_distance_error(start, index + 1))
        ferr = np.row_stack(self.traj.get_friction_error(start, index + 1))
        # Calculate model weights
        dweights = np.linalg.lstsq(Kd, derr, rcond=None)[0]
        fc_weights = np.linalg.lstsq(Kf, ferr, rcond=None)[0]
        fc_weights = self._variables_to_friction_weights(fc_weights, forces)
        model.add_samples(cpts, dweights, fc_weights)
        return model

    def update_trajectory(self, t, result):
        """
            Update the contact estimation trajectory from the problem result
        """
        # Stash the forces, slack variables, etc
        forces = result.GetSolution(self.forces).reshape(self.forces.shape)
        vslacks = result.GetSolution(self.velocities).reshape(self.velocities.shape)
        relax = result.GetSolution(self.feasibilities).reshape(self.feasibilities.shape)
        idx = self.traj.getTimeIndex(t)
        self.traj.set_force(idx, forces[:, -1])
        self.traj.set_dissipation(idx, vslacks[:, -1])
        self.traj.set_feasibility(idx, relax[:, -1])
        # Get the distance and friction weights
        dweights = result.GetSolution(self._distance_weights).reshape(self._distance_weights.shape)
        fweights = result.GetSolution(self._friction_weights).reshape(self._friction_weights.shape)
        # Calculate and update the distance errors
        derr = self._distance_kernel.dot(dweights).reshape((self.traj.num_contacts, self.horizon))
        self.traj.set_distance_error(idx, derr[:, -1])
        # Calculate and update the friction errors
        fc_weights = self._variables_to_friction_weights(fweights, forces)
        fc_err = self._friction_kernel.dot(fc_weights).reshape((self.traj.num_contacts, self.horizon))
        self.traj.set_friction_error(idx, fc_err[:, -1])

    def _variables_to_friction_weights(self, fweights, forces):
        """
            Calculate the friction coefficient weights from solution variables
        """
        fN = forces[:self.traj.num_contacts,:].reshape(-1)
        fc_weights = np.zeros_like(fweights)
        err_index = fN > self.traj._FTOL 
        if np.any(err_index):
            fc_weights[err_index] = fweights[err_index]/fN[err_index]
        return fc_weights

    def get_updated_contact_model(self, result):
        """
            Return a copy of the contact model after updating from the optimization result
        """
        dweights = result.GetSolution(self._distance_weights)
        fc_weights = self._variables_to_friction_weights(result.GetSolution(self._friction_weights), result.GetSolution(self.forces))
        model = copy.deepcopy(self.traj.contact_model)
        cpts = self.traj.get_contacts(self._startptr, self._startptr + self.horizon)
        cpts = np.concatenate(cpts, axis=1)
        model.add_samples(cpts, dweights, fc_weights)    
        return model

    def _clear_program(self):
        """Clears the pointers to the mathematical program and it's decision variables"""
        # Internal variables - program and decision variables
        self._prog = None
        self._distance_weights = None
        self._friction_weights = None
        self._normal_forces = []
        self._friction_forces = []
        self._velocity_slacks = []
        self._relaxation_vars = []
        self._distance_kernel = None
        self._friction_kernel = None
        self._force_cost = None
        self._relax_cost = None
        self._distance_cost = None
        self._friction_cost = None
        self._dist_cstr = []
        self._diss_cstr = []
        self._fric_cstr = []
        self._dyns_cstr = []
        self._fric_pos = []

    def create_estimator(self):
        """
        Create a mathematical program to solve the estimation problem
        """
        self._clear_program()
        self._prog = MathematicalProgram()
        # Determine whether or not we can use the whole horizon
        N = self.traj.num_timesteps
        if N < self.maxhorizon:
            self._startptr = 0
            self.horizon = N
        else:
            self._startptr = N - self.maxhorizon
            self.horizon = self.maxhorizon
        # Setup the distance and friction kernels
        self._distance_kernel, self._friction_kernel = self.traj.getContactKernels(self._startptr, self._startptr + self.horizon)
        # Add the kernel weights
        self._add_kernel_weights()
        self._initialize_weights()
        # Add all the contact constraints
        for index in range(self.horizon):
            self._add_force_variables()
            self._add_dynamics_constraints(index)
            self._add_distance_constraints(index)
            self._add_dissipation_constraints(index)
            self._add_friction_cone_constraints(index)
        # Add the cost functions
        self._add_costs()

    def _add_kernel_weights(self):
        """
        Add the kernel weights as decision variables in the program, and add the associated quadratic costs
        """
        # Create decision variables 
        dvars = self._distance_kernel.shape[1]
        fvars = self._friction_kernel.shape[1]
        self._distance_weights = self._prog.NewContinuousVariables(rows = dvars, name='distance_weights')
        self._friction_weights = self._prog.NewContinuousVariables(rows = fvars, name='friction_weights')
        # Add the quadratic costs
        self._distance_cost = self._prog.AddQuadraticCost(self._distance_kernel * self._distance_cost_weight, np.zeros((dvars,)), self._distance_weights)
        self._distance_cost.evaluator().set_description('Distance Error Cost')
        self._friction_cost = self._prog.AddQuadraticCost(self._friction_kernel * self._friction_cost_weight, np.zeros((fvars,)), self._friction_weights)
        self._friction_cost.evaluator().set_description('Friction Error Cost')

    def _initialize_weights(self):
        """Set the initial guess for the kernel weights"""
        # Initialize the distance kernel weights
        derr = np.concatenate(self.traj.get_distance_error(self._startptr, self._startptr + self.horizon), axis=0)
        dweights = np.linalg.lstsq(self._distance_kernel, derr, rcond=None)[0]
        self._prog.SetInitialGuess(self._distance_weights, dweights)
        # Initialize the friction kernel weights
        ferr = np.concatenate(self.traj.get_friction_error(self._startptr, self._startptr + self.horizon), axis=0)
        force = np.column_stack(self.traj.get_forces(self._startptr, self._startptr + self.horizon))
        fN = force[:self.traj.num_contacts, :].reshape(-1)
        fweights = np.linalg.lstsq(self._friction_kernel, ferr * fN, rcond = None)[0]
        self._prog.SetInitialGuess(self._friction_weights, fweights)

    def _add_force_variables(self):
        """
        Add forces and velocity slacks as decision variables to the program
        """
        # Parameters
        nc = self.traj.num_contacts
        nf = self.traj.num_friction
        # Add the forces, etc
        self._normal_forces.append(self._prog.NewContinuousVariables(rows=nc, name='normal_forces'))
        self._friction_forces.append(self._prog.NewContinuousVariables(rows=nf, name='friction_forces'))
        self._velocity_slacks.append(self._prog.NewContinuousVariables(rows=nc, name='velocity_slacks'))
        # Add the relaxation variables
        self._relaxation_vars.append(self._prog.NewContinuousVariables(rows=1, name='relaxation'))

    def _add_costs(self):
        """
        Add costs on the reaction forces and relaxation variables
        """
        # Force cost
        all_forces = np.ravel(np.row_stack([self.forces, self.velocities]))
        weight = self._force_cost_weight * np.ones(all_forces.shape)
        self._force_cost = self._prog.AddLinearCost(weight, all_forces)
        self._force_cost.evaluator().set_description('Force Cost')
        # Relaxation cost 
        all_relax = np.concatenate(self._relaxation_vars, axis=0)
        r_weights = self._relax_cost_weight * np.eye(all_relax.shape[0])
        r_ref = np.zeros((all_relax.shape[0],))
        self._relax_cost = self._prog.AddQuadraticErrorCost(r_weights, r_ref, all_relax)
        self._relax_cost.evaluator().set_description('Relaxation Cost')
        # Bounding box constraint on the relaxations
        self._prog.AddBoundingBoxConstraint(r_ref, np.full(r_ref.shape, np.inf), all_relax).evaluator().set_description("Relaxation Nonnegativity")

    def _add_dynamics_constraints(self, index):
        """Add and initialize the linear dynamics constraints for force estimation"""
        A, b = self.traj.getDynamicsConstraint(self._startptr + index)
        self._dyns_cstr.append(cstr.RelaxedLinearConstraint(A, b))
        force  = np.concatenate([self._normal_forces[-1], self._friction_forces[-1]], axis=0)
        self._dyns_cstr[-1].addToProgram(self._prog, force, relax=self._relaxation_vars[-1])
        # Set the initial guess
        self._prog.SetInitialGuess(force, self.traj.getForceGuess(self._startptr + index))
        # Set the initial guess for the relaxation variables
        self._prog.SetInitialGuess(self._relaxation_vars[-1], self.traj.getFeasibilityGuess(self._startptr + index))

    def _add_distance_constraints(self, index):
        """Add and initialize the semiparametric distance constraints"""
        kstart, kstop = self.kernelslice(index)
        dist = self.traj.getDistanceConstraint(self._startptr + index)
        self._dist_cstr.append(self.lcp(A = self._distance_kernel[kstart:kstop, :], c = dist))
        self._dist_cstr[-1].set_description('distance')
        self._dist_cstr[-1].addToProgram(self._prog, self._distance_weights, self._normal_forces[-1], rvar=self._relaxation_vars[-1])
        self._dist_cstr[-1].initializeSlackVariables()

    def _add_dissipation_constraints(self, index):
        """Add and initialize the parametric maximum dissipation constraints"""
        D, v = self.traj.getDissipationConstraint(self._startptr + index)
        self._diss_cstr.append(self.lcp(D, v))
        self._diss_cstr[-1].set_description('dissipation')
        self._diss_cstr[-1].addToProgram(self._prog, self._velocity_slacks[-1], self._friction_forces[-1], rvar=self._relaxation_vars[-1])
        # Set the initial guess for the slack variables
        self._prog.SetInitialGuess(self._velocity_slacks[-1], self.traj.getDissipationGuess(self._startptr + index))
        self._diss_cstr[-1].initializeSlackVariables()
        
    def _add_friction_cone_constraints(self, index):
        """Add and initialize the friction cone constraints"""
        # Get the appropriate kernel slice
        kstart, kstop = self.kernelslice(index)
        # Get the friction cone constraints
        D, mu = self.traj.getFrictionConstraint(self._startptr+index)
        A = np.concatenate([np.diag(mu), -D, self._friction_kernel[kstart:kstop, :]], axis=1)
        self._fric_cstr.append(self.lcp(A, np.zeros((A.shape[0], ))))
        self._fric_cstr[-1].set_description('friction_cone')
        xvars = np.concatenate([self._normal_forces[-1], self._friction_forces[-1], self._friction_weights], axis=0)
        zvars = self._velocity_slacks[-1]
        self._fric_cstr[-1].addToProgram(self._prog, xvars, zvars, rvar=self._relaxation_vars[-1])
        # Add a linear constraint to enforce the friction coefficient be nonnegative
        B = np.concatenate([np.diag(mu), self._friction_kernel[kstart:kstop,:]], axis=1) 
        cstr = self._prog.AddLinearConstraint(B, 
                                    lb=np.zeros((self.traj.num_contacts, )), 
                                    ub=np.full((self.traj.num_contacts,), np.inf), 
                                    vars=np.concatenate([self._normal_forces[-1], self._friction_weights], axis=0))
        cstr.evaluator().set_description('friction_coeff_nonnegativity')
        self._fric_pos.append(cstr)
        # Initialize the slack variables in the friction cone constraint
        self._fric_cstr[-1].initializeSlackVariables()

    def kernelslice(self, index):
        """Map the current index to start and stop indices for the kernel matrices"""
        nc = self.traj.num_contacts
        return nc * index, nc * (index + 1)

    @property
    def relaxedcost(self):
        return self._relax_cost_weight

    @relaxedcost.setter
    def relaxedcost(self, val):
        assert isinstance(val, (int, float)) and val >=0, f"val must be a nonnegative float or a nonnegative int"
        self._relax_cost_weight = val
        if self._relax_cost is not None:
            # Update the relaxation costs in the program
            nrelax = len(self._relaxation_vars)
            self._relax_cost.evaluator().UpdateCoefficients(val*np.eye(nrelax), np.zeros((nrelax,)))

    @property
    def forcecost(self):
        return self._force_cost_weight

    @forcecost.setter
    def forcecost(self, val):
        assert isinstance(val, (int, float)) and val >= 0, "val must be a nonnegative float or nonnegative int"
        self._force_cost_weight = val
        # Update the force cost in the program
        if self._force_cost is not None:
            allforceshape = np.ravel(self.forces).shape
            self._force_cost.evaluator().UpdateCoefficients(val * np.ones(allforceshape))

    @property
    def distancecost(self):
        return self._distance_cost_weight

    @distancecost.setter
    def distancecost(self, val):
        assert isinstance(val, (int, float)) and val >= 0, 'val must be a nonnegative float or nonnegative int'
        self._distance_cost_weight = val
        # Update cost in the program
        if self._distance_cost is not None:
            self._distance_cost.evaluator().UpdateCoefficients(val * self._distance_kernel, np.zeros((self._distance_kernel.shape[0], )))

    @property
    def frictioncost(self):
        return self._friction_cost_weight

    @frictioncost.setter
    def frictioncost(self, val):
        assert isinstance(val, (int, float)) and val >= 0, 'val must be a nonnegative float or nonnegative int'
        self._friction_cost_weight = val
        # update the cost in the program
        if self._friction_cost is not None:
            self._friction_cost.evaluator().UpdateCoefficients(val * self._friction_kernel, np.zeros((self._friction_kernel.shape[0], )))

    @property
    def forces(self):
        if self._normal_forces is not []:
            fN = np.column_stack(self._normal_forces)
            fT = np.column_stack(self._friction_forces)
            return np.row_stack([fN, fT])
        else:
            return None

    @property
    def velocities(self):
        if self._velocity_slacks is not []:
            return np.column_stack(self._velocity_slacks)
        else:
            return None

    @property
    def feasibilities(self):
        if self._relaxation_vars is not []:
            return np.column_stack(self._relaxation_vars)
        else:
            return None

class ContactModelEstimatorNonlinearFrictionCone(ContactModelEstimator):
    def _add_friction_cone_constraints(self, index):
        """Add and initialize the friction cone constraints"""
        # Get the appropriate kernel slice
        kstart, kstop = self.kernelslice(index)
        # Get the friction cone constraints
        D, mu = self.traj.getFrictionConstraint(self._startptr+index)
        self._fric_cstr.append(SemiparametricFrictionConeConstraint(mu, self._friction_kernel[kstart:kstop, :], D))
        self._fric_cstr[-1].addToProgram(self.prog,
            xvars = np.concatenate([self._friction_weights, self._normal_forces[-1], self._friction_forces[-1]], axis=0),
            zvars = self._velocity_slacks[-1],
            rvars = self._relaxation_vars[-1])

    def _variables_to_friction_weights(self, fweights, forces):
        """
            Calculate the friction coefficient weights from solution variables
        """
        return fweights

class EstimatedContactModelRectifier(OptimizationMixin):
    def __init__(self, esttraj, surf_max=np.inf, fric_max = np.inf):
        super().__init__()
        assert isinstance(esttraj, ContactEstimationTrajectory), "EstimatedContactModelRectifier must be initialized with a ContactEstimationTrajectory object"
        self.traj = esttraj
        self.surf_max = surf_max
        self.fric_max = fric_max
        self._setup()

    def _setup(self):
        """Set up the optimization program"""
        # Store the kernel matrices for the entire problem
        self.Kd, self.Kf = self.traj.getContactKernels(0, self.traj.num_timesteps)
        # Create the contact duplication matrix
        d = np.ones((self.traj.num_contacts,1))
        self.D = np.kron(np.eye(self.traj.num_timesteps, dtype=int), d)  
        # Setup the optimization program - just the constraints
        self._prog = MathematicalProgram()
        self._add_variables()
        self._add_distance_constraints()
        self._add_friction_constraints()
        self._add_relaxation_constraint()
        self._initialize()
        self.costs = []

    def _add_variables(self):
        """Add decision variables to the optimization problem"""
        self.dweights = self.prog.NewContinuousVariables(rows = self.Kd.shape[1], name='distance_weights')
        self.fweights = self.prog.NewContinuousVariables(rows = self.Kf.shape[1], name='friction_weights')
        self.relax = self.prog.NewContinuousVariables(rows = self.traj.num_timesteps, name='feasibilities')

    def _add_distance_constraints(self):
        """Add two linear constraints for the normal distance errors"""
        # Get the distance and the forces
        dist0 = np.concatenate(self.traj._distance_cstr, axis=0)
        fN = np.concatenate([force[:self.traj.num_contacts] for force in self.traj._forces], axis=0)
        # Add the distance nonnegativity constraint
        self.prog.AddLinearConstraint(A = self.Kd, 
                                    lb = -dist0, 
                                    ub = np.full(dist0.shape, self.surf_max) - dist0, 
                                    vars = self.dweights).evaluator().set_description('Distance Nonnegativity')
        # Add the orthogonality constraint
        A = np.concatenate([fN * self.Kd, -self.D], axis=1)
        b = fN * dist0  
        self.prog.AddLinearConstraint(A = A, 
                                    lb = -np.full(dist0.shape[0], np.inf), 
                                    ub = -b, 
                                    vars=np.concatenate([self.dweights, self.relax], axis=0)).evaluator().set_description('Distance Orthogonality')
        
    def _add_friction_constraints(self):
        """Add to linear constraints for the friction cone error"""
        # Get the friction coefficients and friction cone defects
        e = self.traj._D
        b = []
        fN_all = []
        all_mu = []
        for mu, forces in zip(self.traj._friction_cstr, self.traj._forces):
            fN, fT = np.split(forces, [self.traj.num_contacts])
            b.append(mu * fN - e.dot(fT))
            fN_all.append(fN)
            all_mu.append(mu)
        # Setup the nonnegativity constraint
        b = np.concatenate(b, axis=0)
        fN = np.concatenate(fN_all, axis=0)
        self.prog.AddLinearConstraint(A = fN * self.Kf, 
                                    lb = -b, 
                                    ub = np.full(b.shape[0], self.fric_max) - b,
                                    vars = self.fweights).evaluator().set_description('Friction Cone Nonnegativity')
        # Setup the orthogonality constraint
        sV = np.concatenate(self.traj._slacks, axis=0)
        A = np.concatenate([sV * fN * self.Kf, -self.D], axis=1)
        b = sV * b
        self.prog.AddLinearConstraint(A = A,
                                    lb = -np.full(b.shape[0], np.inf),
                                    ub = -b,
                                    vars = np.concatenate([self.fweights, self.relax], axis=0)).evaluator().set_description('Friction Cone Orthogonality')
        # Add the constraint on the friction coefficient
        mu = np.concatenate(all_mu, axis=0)
        self.prog.AddLinearConstraint(A = self.Kf, lb = -mu, ub = np.full(mu.shape[0], np.inf), vars=self.fweights).evaluator().set_description('Friction Coefficient Nonnegativity')

    def _add_relaxation_constraint(self):
        """Add the bounding box constraint on the relaxation variables"""
        f = np.concatenate(self.traj._feasibility, axis=0)
        f = np.expand_dims(f, axis=1)
        relax = np.expand_dims(self.relax, axis=1)
        self.prog.AddBoundingBoxConstraint(np.zeros(f.shape), f, relax).evaluator().set_description('Feasibility Limits')

    def _initialize(self):
        """Initialize the decision variables"""
        derr = np.concatenate(self.traj.get_distance_error(0, self.traj.num_timesteps), axis=0)
        ferr = np.concatenate(self.traj.get_friction_error(0, self.traj.num_timesteps), axis=0)
        relax = np.concatenate(self.traj.get_feasibility(0, self.traj.num_timesteps), axis=0)
        # Convert to kernel weights
        dweight = np.linalg.lstsq(self.Kd, derr, rcond=None)[0]
        fweight = np.linalg.lstsq(self.Kf, ferr, rcond=None)[0]
        # Set initial guess
        self.prog.SetInitialGuess(self.dweights, dweight)
        self.prog.SetInitialGuess(self.fweights, fweight)
        self.prog.SetInitialGuess(self.relax, relax)

    def _add_quadratic_cost(self):
        """Add the quadratic cost terms used in global model optimization"""
        # Create the costs
        distance_cost = self.prog.AddQuadraticErrorCost(self.Kd, np.zeros(self.dweights.shape), self.dweights)
        distance_cost.evaluator().set_description('Quadratic Distance Cost')
        friction_cost = self.prog.AddQuadraticErrorCost(self.Kf, np.zeros(self.fweights.shape), self.fweights)
        friction_cost.evaluator().set_description('Quadratic Friction Cost')
        # Store the cost terms in case we need to remove them later
        self.costs.extend([distance_cost, friction_cost])

    def _add_linear_cost(self, maximize=False):
        """Add the linear cost terms used in ambiguity set optimization"""
        coeff = (-1)**maximize
        # Create the costs
        distance_cost = self.prog.AddLinearCost(coeff * np.sum(self.Kd, axis=0), self.dweights)
        friction_cost = self.prog.AddLinearCost(coeff * np.sum(self.Kf, axis=0), self.fweights)
        distance_cost.evaluator().set_description('Linear Distance Cost')
        friction_cost.evaluator().set_description('Linear Friction Cost')
        # Keep the bindings in case we need to remove them later
        self.costs.extend([distance_cost, friction_cost])

    def _clear_costs(self):
        """Remove all cost terms from the optimization problem"""
        for cost in self.costs:
            self.prog.RemoveCost(cost)

    def solve_ambiguity(self):
        """
        Solve the ambiguity set optimization
        
        """
        # Lower bound optimization
        self._clear_costs()
        self._add_linear_cost(maximize=False)
        lb = self.solve()
        # Upper bound optimization
        self._clear_costs()
        self._add_linear_cost(maximize=True)
        ub = self.solve()
        return lb, ub

    def solve_global_model(self):
        """Solve the global model optimization problem"""
        self._clear_costs()
        self._add_quadratic_cost()
        return self.solve()

    def solve_global_model_with_ambiguity(self):
        """
        Solve both the global model optimization and the ambiguity model optimization
        
        Returns a SemiparametricContactModelWithAmbiguity
        """
        global_model = self.solve_global_model()
        if not global_model.is_success():
            warnings.warn('Failed to solve global contact model optimization. Results may be inaccurate')
        lb, ub = self.solve_ambiguity()
        if not lb.is_success():
            warnings.warn('Failed to solve lower bound contact model optimization. Results may be inaccurate')
        if not ub.is_success():
            warnings.warn('Failed to solve upper bound contact model. Results may be inaccurate')
        # Get and check the results
        surf_global = global_model.GetSolution(self.dweights)
        fric_global = global_model.GetSolution(self.fweights)
        surf_lb = lb.GetSolution(self.dweights)
        surf_ub = ub.GetSolution(self.dweights)
        fric_lb = lb.GetSolution(self.fweights)
        fric_ub = ub.GetSolution(self.fweights)
        # Create the model with ambiguity information
        model = copy.deepcopy(self.traj.contact_model)
        model = model.toSemiparametricModelWithAmbiguity()
        cpts = np.concatenate(self.traj.get_contacts(0, self.traj.num_timesteps), axis=1)
        model.add_samples(cpts, surf_global, fric_global)
        model.set_upper_bound(surf_ub, fric_ub)
        model.set_lower_bound(surf_lb, fric_lb)
        return model

class ContactEstimationPlotter():
    def __init__(self, traj):
        assert isinstance(traj, ContactEstimationTrajectory), 'ContactEstimationPlotter requires a ContactEstimationTrajectory object'
        self.traj = traj

    def plot(self, show=True, savename=None):
        self.plot_forces(show=False, savename=savename)
        # Plot the velocities and feasibilities in one graph
        _, axs1 = plt.subplots(2, 1)
        self.plot_velocities(axs1[0], show=False, savename=None)
        self.plot_feasibilities(axs1[1], show=False, savename=utils.append_filename(savename, '_feasibility'))
        # Plot the surface errors and friction errors in another graph
        _, axs2 = plt.subplots(2, 1)
        self.plot_surface_errors(axs2[0], show=False, savename=None)
        self.plot_friction_errors(axs2[1], show=False, savename=utils.append_filename(savename, '_errors'))
        if show:
            plt.show()


    def plot_forces(self, show=False, savename=None):
        """
        Plot the force trajectories in ContactEstimationTrajectory
        """
        f = np.column_stack(self.traj._forces)
        ftraj = pp.ZeroOrderHold(self.traj._time, f)
        # Use the plant's native plot_forces command for this one
        return self.traj._plant.plot_force_trajectory(ftraj, show=show, savename=utils.append_filename(savename, '_reactions'))

    @deco.showable_fig
    @deco.saveable_fig
    def plot_feasibilities(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots(1,1)
        else:
            # Get the figure from the current axis
            plt.sca(axs)
            fig = plt.gcf()
        t = np.row_stack(self.traj._time)
        f = np.row_stack(self.traj._feasibility)
        axs.plot(t, f, linewidth=1.5, color='black')
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Feasibility')
        axs.set_yscale('symlog', linthresh=1e-6)
        axs.grid(True)
        fig.tight_layout()
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_velocities(self, axs=None):
        """
        Plot the maximum tangential sliding velocity variables
        """
        if axs is None:
            fig, axs = plt.subplots(1, 1)
        else:
            # Get the figure from the current axis
            plt.sca(axs)
            fig = plt.gcf()
        t = np.row_stack(self.traj._time)
        v = np.column_stack(self.traj._slacks)
        for n in range(self.traj.num_contacts):
            axs.plot(t, v[n,:], linewidth=1.5)
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Maximum sliding velocity')
        fig.tight_layout()
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_surface_errors(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots(1,1)
        else:
            plt.sca(axs)
            fig = plt.gcf()
        s_err = np.column_stack(self.traj._distance_error)
        t = np.row_stack(self.traj._time)
        for n in range(self.traj.num_contacts):
            axs.plot(t, s_err[n,:], linewidth=1.5)
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Distance Error')
        fig.tight_layout()
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_friction_errors(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots(1,1)
        else:
            plt.sca(axs)
            fig = plt.gcf()
        f_err = np.column_stack(self.traj._friction_error)
        t = np.row_stack(self.traj._time)
        for n in range(self.traj.num_contacts):
            axs.plot(t, f_err[n,:], linewidth=1.5)
        axs.set_xlabel('Time (s)')
        axs.set_ylabel('Friction error')
        fig.tight_layout()
        return fig, axs

if __name__ == '__main__':
    print("Hello from contactestimator!")