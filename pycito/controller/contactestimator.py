"""
Contact model estimation

Luke Drnach
February 14, 2022
"""
#TODO: Unittesting
#TODO: Refactor ContactModelEstimator to reuse and update the program
#TODO: Implement 'contact model rectifier' to calculate the global contact model offline - this is a QP
#TODO: Implement 'ambiguity set optimization' to calculate bounds on contact model - this is a LP
 
import numpy as np

from pydrake.all import MathematicalProgram, SnoptSolver

import pycito.trajopt.constraints as cstr
import pycito.trajopt.complementarity as cp
import pycito.controller.mlcp as mlcp
from pycito.systems.contactmodel import SemiparametricContactModel

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
            time: the timestamp to append

        Requirements:
            time must be a float, and time must be greater than the current value at the end of self._time - i.e. the overall sequence of timesteps must be monotonically increasing
        """
        assert isinstance(time, float), f'timestamp must be a float'
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

    @property
    def num_timesteps(self):
        return len(self._time)

class ContactEstimationTrajectory(ContactTrajectory):
    def __init__(self, plant, initial_state):
        super(ContactEstimationTrajectory, self).__init__()
        self._plant = plant
        self._context = plant.multibody.CreateDefaultContext()
        # Store the last state for calculating dynamics
        self._last_state = initial_state
        # Setup the constraint parameter lists
        self._dynamics_cstr = []
        self._distance_cstr = []
        self._dissipation_cstr = []
        self._friction_cstr = []
        # Store the duplication matrix
        self._D = self._plant.duplicator_matrix()
        # Store the terrain kernel matrices
        self.contact_model = plant.terrain
        assert isinstance(self.contact_model, SemiparametricContactModel), f"plant does not contain a semiparametric contact model"

    def append_sample(self, time, state, control):
        """
        Append a new point to the trajectory
        Also appends new parameters for the dynamics, normal distance, dissipation, and friction coefficient
        """
        # Add the dynamics constraint first, as this manipulates the context
        self._append_dynamics(time, state, control)
        # Reset the context, and add the contact points
        self._plant.multibody.SetPositionsAndVelocities(self._context, state)
        cpts = self._plant.get_contact_points(self._context)
        self.add_contact_sample(time, np.hstack(cpts))
        # Also add the distance, dissipation, and friction constraints
        self._append_distance()
        self._append_dissipation()
        self._append_friction()

    def _append_dynamics(self, time, state, control):
        """
        Add a set of linear system parameters to evaluate the dynamics defect 
        """
        dt = time - self._time[-1]
        # Get the Jacobian matrix - the "A" parameter
        self._plant.multibody.SetPositionsAndVelocities(self._context, state)
        Jn, Jt = self._plant.GetContactJacobians(self._context)
        A = dt*np.concatenate([Jn.T, Jt.T], axis=0)
        # Now get the dynamics defect
        if self._plant.has_joint_limits:
            forces = np.zeros((A.shape[1] + self._plant.joint_limit_jacobian().shape[1], ))
        else:
            forces = np.zeros((A.shape[1], ))
        b = cstr.BackwardEulerDynamicsConstraint.eval(self._plant, self._context, dt, self._last_state, state, control, forces)
        # Append - wrap this in it's own constraint
        self._dynamics_cstr.append((A, -b))
        # Save the state for the the next call to append_dynamics
        self._last_state = state

    def _append_distance(self):
        """
        Add a set of linear system parameters to evaluate the normal distance defect
        """
        # Get the distance vector
        b = self._plant.GetNormalDistances(self._context)
        self._distance_cstr.append(b)

    def _append_dissipation(self):
        """
        Add a set of linear system parameters to evaluate the dissipation defect
        """
        # Get the tangent jacobian
        _, Jt = self.plant.GetContactJacobians(self._context)
        v = self.plant.multibody.GetVelocities(self._context)
        b = Jt.dot(v)
        self._dissipation_cstr.append(b)

    def _append_friction(self):
        """
        Add a set of linear system parameters to evaluate the friction cone defect
        """
        # Append the friction coefficients
        mu = self.plant.GetFrictionCoefficients(self._context)
        self._friction_cstr.append(np.atleast_1d(mu))

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
        return self._D.transpose(), self._dissipation_cstr[index]

    def getFrictionConstraint(self, index):
        """
        Return the friction cone constraint at the specified index
        """
        assert index >= 0  and index < len(self._friction_cstr), "index out of bounds"
        return self._D, self._friction_cstr[index]

    def getForceGuess(self, index):
        """
        Return the initial guess for the forces at the specified index
        """
        if index < len(self._forces):
            return self._forces[index]
        else:
            A, b = self.getDynamicsConstraint(index)
            f = np.linalg.solve(A, b)
            f[f < 0] = 0
            return f

    def getDissipationGuess(self, index):
        """
        Return the initial guess for the dissipation slacks at the specified index
        """
        if index < len(self._slacks):
            return self._slacks[index]
        else:
            D, v = self.getDissipationConstraint(index)
            return np.max(-v)*np.ones((D.shape[1], 1))

    def getFeasibilityGuess(self, index):
        """
        Return an initial guess for the feasibility relaxation variable
        """
        if index < len(self._feasibility):
            return self._feasibility[index]
        else:
            return np.ones((1,))

    def getContactKernels(self, start, stop=None):
        # Get the contact points
        cpts = np.concatenate(self.get_contacts(start, stop), axis=1)
        # Calculate the surface and friction kernel matrices
        return self.contact_model.surface_kernel(cpts), self.contact_model.friction_kernel(cpts)

    @property
    def num_contacts(self):
        return self._plant.num_contacts

    @property
    def num_friction(self):
        return self._plant.num_friction

class ContactModelEstimator():
    def __init__(self, esttraj, horizon):
        """
        Arguments:
            esttraj: a ContactEstimationTrajectory object
            horizon: a scalar indicating the reverse horizon to use for estimation
        """
        self.traj = esttraj
        self.maxhorizon = horizon
        self._clear_program()
        # Cost weights
        self._relax_cost_weight = 1.
        self._force_cost_weight = 1.
        # Solver details
        self.solveroptions = {}
        self._solver = SnoptSolver()

    def setSolverOptions(self, optionsdict = {}):
        # Store the solver options
        for key, value in optionsdict.items():
            self.solveroptions[key] = value

    def estimate_contact(self, t, x, u):
        # Append new samples
        self.traj.append_sample(t, x, u)
        # Create contact estimation problem 
        print(f"Creating contact estimation problem")
        self.create_estimator()
        print(f"Solving contact estimation")
        result = self.solve()
        # Stash the forces, slack variables, etc
        forces = result.GetSolution(self.forces)
        vslacks = result.GetSolution(self.velocities)
        relax = result.GetSolution(np.concatenate(self._relaxation_vars, axis=0))
        self.traj.add_force(t, forces[:, -1])
        self.traj.add_velocity(t, vslacks[:, -1])
        self.traj.add_feasibility(t, relax[-1])
        # Get the distance and friction weights
        dweights = result.GetSolution(self._distance_weights)
        fweights = result.GetSolution(self._friction_weights)
        # TODO: Return an updated contact model
        return dweights, fweights

    def solve(self):
        """Solves the Contact Model Estimation program"""
        # Update solver options
        for key, value in self.solveroptions.items():
            self.prog.SetSolverOption(self._solver.solver_id(), key, value)
        # Solve the estimation problem
        return self._solver.Solve(self._prog)

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

    def create_estimator(self):
        """
        Create a mathematical program to solve the estimation problem
        """
        self._clear_program()
        self._prog = MathematicalProgram()
        # Determine whether or not we can use the whole horizon
        N = self.traj.num_timesteps
        if N - self.maxhorizon < 0:
            self._startptr = 0
            self.horizon = N
        else:
            self._startptr = N - self.maxhorizon
            self.horizon = self.maxhorizon
        # Setup the distance and friction kernels
        self._distance_kernel, self._friction_kernel = self.traj.getContactKernels(self._startptr, self._startptr + self.horizon)
        # Add the kernel weights
        self._add_kernel_weights()
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
        self._prog.AddQuadraticCost(self._distance_kernel, np.zeros((dvars,)), self._distance_weights).evaluator().set_description('Distance Error Cost')
        self._prog.AddQuadraticCost(self._friction_kernel, np.zeros((fvars,)), self._friction_weights).evaluator().set_description('Friction Error Cost')
        # Initialize the kernel weights
        self._prog.SetInitialGuess(self._friction_weights, np.zeros(self._friction_weights.shape))
        distances = np.concatenate(self.traj._distance_cstr[self._startptr:self._startptr + self._horizon], axis=0)
        guess = np.linalg.solve(self._distance_kernel, -distances)
        self._prog.SetInitialGuess(self._distance_weights, guess)

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
        all_forces = np.ravel(self.forces)
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
        A, b = self.traj.getDynamicsConstraint(self._startprt + index)
        dyn_cstr = cstr.RelaxedLinearConstraint(A, b)
        force  = np.concatenate([self._normal_forces[-1], self._friction_forces[-1]], axis=0)
        dyn_cstr.addToProgram(self._prog, force, relax=self._relaxation_vars[-1])
        # Set the initial guess
        self._prog.SetInitialGuess(force, self.traj.getForceGuess(self._startptr + index))
        # Set the initial guess for the relaxation variables
        self._prog.SetInitialGuess(self._relaxation_vars[-1], self.traj.getFeasibilityGuess(self._startptr + index))

    def _add_distance_constraints(self, index):
        """Add and initialize the semiparametric distance constraints"""
        kstart, kstop = self.kernelslice(index)
        dist = self.traj.getDistanceConstraint(self._startptr + index)
        dist_cstr = mlcp.VariableRelaxedPseudoLinearComplementarityConstraint(A = self._distance_kernel[kstart:kstop, :], c = dist)
        dist_cstr.addToProgram(self._prog, self._distance_weights, self._normal_forces[-1], rvar=self._relaxation_vars[-1])
        dist_cstr.initializeSlackVariables()

    def _add_dissipation_constraints(self, index):
        """Add and initialize the parametric maximum dissipation constraints"""
        D, v = self.traj.getDissipationConstraint(self._startptr + index)
        diss_cstr = mlcp.VariableRelaxedPseudoLinearComplementarityConstraint(D, v)
        diss_cstr.addToProgram(self._prog, self._velocity_slacks[-1], self._friction_forces[-1], rvar=self._relaxation_vars[-1])
        diss_cstr.initializeSlackVariables()
        # Set the initial guess for the slack variables
        self._prog.SetInitialGuess(self._velocity_slacks[-1], self.traj.getDissipationGuess(self._startptr + index))
        
    def _add_friction_cone_constraints(self, index):
        """Add and initialize the friction cone constraints"""
        # Get the appropriate kernel slice
        kstart, kstop = self.kernelslice(index)
        #Store the friction cone constraints to set the weights accordingly
        D, mu = self.traj.getFrictionConeConstraint(self._startptr + index)
        fc_cstr = SemiparametricFrictionConeConstraint(mu, self._friction_kernel[kstart:kstop, :], D)
        xvars = np.concatenate([self._friction_weights, self._normal_forces[-1], self._friction_forces[-1]], axis=0)
        zvars = self._velocity_slacks[-1]
        fc_cstr.addToProgram(self._prog, xvars, zvars, rvars=self._relaxation_vars[-1])    

    def kernelslice(self, index):
        """Map the current index to start and stop indices for the kernel matrices"""
        nc = self.traj.num_contacts
        return nc * index, nc * (index + 1)

    @property
    def relaxedcost(self):
        return self._relax_cost_weight

    @relaxedcost.setter
    def relaxedcost(self, val):
        assert isinstance(val, [int, float]) and val >=0, f"val must be a nonnegative float or a nonnegative int"
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
        assert isinstance(val, [int, float]) and val >= 0, "val must be a nonnegative float or nonnegative int"
        self._force_cost_weight = val
        # Update the force cost in the program
        if self._force_cost is not None:
            allforceshape = np.ravel(self.forces).shape
            self._force_cost.evaluator().UpdateCoefficients(val * np.ones(allforceshape))

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

if __name__ == '__main__':
    print("Hello from contactestimator!")