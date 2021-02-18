"""
contactimplicit: Implements Contact Implicit Trajectory Optimization using Backward Euler Integration
    Partially integrated with pyDrake
    contains pyDrake's MathematicalProgram to formulate and solve nonlinear programs
    uses pyDrake's MultibodyPlant to represent rigid body dynamics
Luke Drnach
October 5, 2020
"""
import numpy as np 
from pydrake.all import MathematicalProgram, PiecewisePolynomial
from pydrake.autodiffutils import AutoDiffXd
from pydrake.multibody.tree import MultibodyForces_
from utilities import MathProgIterationPrinter

class ContactImplicitDirectTranscription():
    """
    Implements contact-implicit trajectory optimization using Direct Transcription
    """
    def __init__(self, plant, context, num_time_samples, minimum_timestep, maximum_timestep):
        """
        Create MathematicalProgram with decision variables and add constraints for the rigid body dynamics, contact conditions, and joint limit constraints

            Arguments:
                plant: a TimeSteppingMultibodyPlant model
                context: a Context for the MultibodyPlant in TimeSteppingMultibodyPlant
                num_time_samples: (int) the number of knot points to use
                minimum_timestep: (float) the minimum timestep between knot points
                maximum_timestep: (float) the maximum timestep between knot points
        """
        # Store parameters
        self.plant_f = plant
        self.context_f = context
        self.num_time_samples = num_time_samples
        self.minimum_timestep = minimum_timestep
        self.maximum_timestep = maximum_timestep
        # Create a copy of the plant and context with scalar type AutoDiffXd
        self.plant_f.multibody.SetDefaultContext(context)
        self.plant_ad = self.plant_f.toAutoDiffXd()       
        self.context_ad = self.plant_ad.multibody.CreateDefaultContext()
        # Create MultibodyForces
        MBF = MultibodyForces_[float]
        self.mbf_f = MBF(self.plant_f.multibody)
        MBF_AD = MultibodyForces_[AutoDiffXd]
        self.mbf_ad = MBF_AD(self.plant_ad.multibody)
        # Create the mathematical program
        self.prog = MathematicalProgram()
        # Check for floating DOF
        self._check_floating_dof()
        # Add decision variables to the program
        self._add_decision_variables()
        # Add dynamic constraints 
        self._add_dynamic_constraints()
        # Add contact constraints
        self._add_contact_constraints()
        # Initialize the timesteps
        self._set_initial_timesteps()

    def _check_floating_dof(self):

        # Get the floating bodies
        floating = self.plant_f.multibody.GetFloatingBaseBodies()
        self.floating_pos = []
        self.floating_vel = []
        while len(floating) > 0:
            body = self.plant_f.multibody.get_body(floating.pop())
            if body.has_quaternion_dofs():
                self.floating_pos.append(body.floating_positions_start())
                self.floating_vel.append(body.floating_velocities_start())

    def _add_decision_variables(self):
        """
            adds the decision variables for timesteps, states, controls, reaction forces,
            and joint limits to the mathematical program, but does not initialize the 
            values of the decision variables. Store decision variable lists

            addDecisionVariables is called during object construction
        """
        # Add time variables to the program
        self.h = self.prog.NewContinuousVariables(rows=self.num_time_samples-1, cols=1, name='h')
        # Add state variables to the program
        nX = self.plant_ad.multibody.num_positions() + self.plant_ad.multibody.num_velocities()
        self.x = self.prog.NewContinuousVariables(rows=nX, cols=self.num_time_samples, name='x')
        # Add control variables to the program
        nU = self.plant_ad.multibody.num_actuators()
        self.u = self.prog.NewContinuousVariables(rows=nU, cols=self.num_time_samples, name='u')
        # Add reaction force variables to the program
        self.numN = self.plant_f.num_contacts()
        self.numT = self.plant_ad.num_friction()
        self.l = self.prog.NewContinuousVariables(rows=2*self.numN+self.numT, cols=self.num_time_samples, name='l')
        # store a matrix for organizing the friction forces
        self._e = self.plant_ad.duplicator_matrix()

        # And joint limit variables to the program
        qlow = self.plant_ad.multibody.GetPositionLowerLimits()
        # Assume that the joint limits be two-sided
        self.Jl = self.plant_ad.joint_limit_jacobian()
        if self.Jl is not None:
            qlow = self.plant_ad.multibody.GetPositionLowerLimits()
            self._liminds = np.isfinite(qlow)
            nJL = sum(self._liminds)
            self.jl = self.prog.NewContinuousVariables(rows = 2*nJL, cols=self.num_time_samples, name="jl")
        else:
            self.jl = False
            
    def _add_dynamic_constraints(self):
        """Add constraints to enforce rigid body dynamics and joint limits"""
        # Check for joint limits first
        if self.Jl is not None:
            # Create the joint limit constraint
            self.joint_limit_cstr = NonlinearComplementarityFcn(self._joint_limit, xdim=self.x.shape[0], zdim=self.jl.shape[0], slack=0)
            for n in range(0, self.num_time_samples-1):
                # Add timestep constraints
                self.prog.AddBoundingBoxConstraint(self.minimum_timestep, self.maximum_timestep, self.h[n,:]).evaluator().set_description('TimestepConstraint')
                # Add dynamics constraints
                self.prog.AddConstraint(self._backward_dynamics, 
                            lb=np.zeros(shape=(self.x.shape[0], 1)),
                            ub=np.zeros(shape=(self.x.shape[0], 1)),
                            vars=np.concatenate((self.h[n,:], self.x[:,n], self.x[:,n+1], self.u[:,n], self.l[:,n+1], self.jl[:,n+1]), axis=0),
                            description="dynamics")
                # Add joint limit constraints
                self.prog.AddConstraint(self.joint_limit_cstr,
                        lb=self.joint_limit_cstr.lower_bound(),
                        ub=self.joint_limit_cstr.upper_bound(),
                        vars=np.concatenate((self.x[:,n+1], self.jl[:,n+1]), axis=0),
                        description="joint_limits")
        else:
            for n in range(0, self.num_time_samples-1):
                # Add timestep constraints
                self.prog.AddBoundingBoxConstraint(self.minimum_timestep, self.maximum_timestep, self.h[n,:]).evaluator().set_description('TimestepConstraint')
                # Add dynamics as constraints 
                self.prog.AddConstraint(self._backward_dynamics, 
                            lb=np.zeros(shape=(self.x.shape[0], 1)),
                            ub=np.zeros(shape=(self.x.shape[0], 1)),
                            vars=np.concatenate((self.h[n,:], self.x[:,n], self.x[:,n+1], self.u[:,n], self.l[:,n+1]), axis=0),
                            description="dynamics")  
           
    def _add_contact_constraints(self):
        """ Add complementarity constraints for contact to the optimization problem"""
        # At each knot point, add constraints for normal distance, sliding velocity, and friction cone
        self.distance_cstr = NonlinearComplementarityFcn(self._normal_distance,
                                                    xdim = self.x.shape[0],
                                                    zdim = self.numN,
                                                    slack = 0.)
        self.sliding_cstr = NonlinearComplementarityFcn(self._sliding_velocity,
                                                    xdim = self.x.shape[0] + self.numN,
                                                    zdim = self.numT,
                                                    slack = 0.)
        self.friccone_cstr = NonlinearComplementarityFcn(self._friction_cone, 
                                                    xdim = self.x.shape[0] + self.numN + self.numT,
                                                    zdim = self.numN,
                                                    slack = 0.)
        for n in range(0, self.num_time_samples):
            # Add complementarity constraints for contact
            self.prog.AddConstraint(self.distance_cstr, 
                        lb=self.distance_cstr.lower_bound(),
                        ub=self.distance_cstr.upper_bound(),
                        vars=np.concatenate((self.x[:,n], self.l[0:self.numN,n]), axis=0),
                        description="normal_distance")
            # Sliding velocity constraint 
            self.prog.AddConstraint(self.sliding_cstr,
                        lb=self.sliding_cstr.lower_bound(),
                        ub=self.sliding_cstr.upper_bound(),
                        vars=np.concatenate((self.x[:,n], self.l[self.numN+self.numT:,n], self.l[self.numN:self.numN+self.numT,n]), axis=0),
                        description="sliding_velocity")
            # Friction cone constraint
            self.prog.AddConstraint(self.friccone_cstr, 
                        lb=self.friccone_cstr.lower_bound(),
                        ub=self.friccone_cstr.upper_bound(),
                        vars=np.concatenate((self.x[:,n], self.l[:,n]), axis=0),
                        description="friction_cone")

    def _backward_dynamics(self, z):  
        """
        backward_dynamics: Backward Euler integration of the dynamics constraints
        Decision variables are passed in through a list in the order:
            z = [h, x1, x2, u, l, jl]
        Returns the dynamics defect, evaluated using Backward Euler Integration. 
        """
        #NOTE: Cannot use MultibodyForces.mutable_generalized_forces with AutodiffXd. Numpy throws an exception
        plant, context, mbf = self._autodiff_or_float(z)
        # Split the variables from the decision variables
        ind = np.cumsum([self.h.shape[1], self.x.shape[0], self.x.shape[0], self.u.shape[0]])
        h, x1, x2, u, l = np.split(z, ind)
        # Split configuration and velocity from state
        q1, v1 = np.split(x1,[plant.multibody.num_positions()])
        q2, v2 = np.split(x2,[plant.multibody.num_positions()])
        # Discretize generalized acceleration
        dv = (v2 - v1)/h
        # Update the context
        plant.multibody.SetPositionsAndVelocities(context, x2)
        # Set mutlibodyForces to zero
        mbf.SetZero()
        # calculate generalized forces
        B = plant.multibody.MakeActuationMatrix()
        forces = B.dot(u)
        # Gravity
        forces[:] = forces[:] + plant.multibody.CalcGravityGeneralizedForces(context)
        # Joint limits
        if self.Jl is not None:
            l, jl = np.split(l, [self.l.shape[0]])
            forces[:] = forces[:] + self.Jl.dot(jl)
        # Ground reaction forces
        Jn, Jt = plant.GetContactJacobians(context)
        J = np.concatenate((Jn, Jt), axis=0)
        forces[:] = forces[:] + J.transpose().dot(l[0:self.numN + self.numT])
        # Do inverse dynamics
        fv = plant.multibody.CalcInverseDynamics(context, dv, mbf) - forces
        # Calc position residual from velocity
        dq2 = plant.multibody.MapVelocityToQDot(context, v2)
        fq = q2 - q1 - h*dq2
        # Check for floating bodies / quaternion variables
        if len(self.floating_pos) > 0:
            for pidx, vidx in zip(self.floating_pos, self.floating_vel):
                fq[pidx:pidx+4] = q2[pidx:pidx+4] - integrate_quaternion(x1[pidx:pidx+4],x1[vidx:vidx+3],h.item())
        # Return dynamics defects
        return np.concatenate((fq, fv), axis=0)
    
    # Complementarity Constraint functions for Contact
    def _normal_distance(self, state):
        """
        Return normal distances to terrain
        
        Arguments:
            The decision variable list:
                vars = [state, normal_forces]
        """
        # Check if the decision variables are floats
        plant, context, _ = self._autodiff_or_float(state)
        # Calculate the normal distance
        plant.multibody.SetPositionsAndVelocities(context, state)    
        return plant.GetNormalDistances(context)

    def _sliding_velocity(self, vars):
        """
        Complementarity constraint between the relative sliding velocity and the tangential reaction forces

        Arguments:
            The decision variable list:
                vars = [state, velocity_slacks]
        """
        plant, context, _ = self._autodiff_or_float(vars)
        # Split variables from the decision list
        x, gam = np.split(vars, [self.x.shape[0]])
        # Get the velocity, and convert to qdot
        _, v = np.split(x, [plant.multibody.num_positions()])
        plant.multibody.SetPositionsAndVelocities(context, x)
        # Get the contact Jacobian
        _, Jt = plant.GetContactJacobians(context)
        # Match sliding slacks to sliding velocities
        return self._e.transpose().dot(gam) + Jt.dot(v)

    def _friction_cone(self, vars):
        """
        Complementarity constraint between the relative sliding velocity and the friction cone

        Arguments:
            The decision variable list is stored as :
                vars = [state,normal_forces, friction_forces]
        """
        plant, context, _ = self._autodiff_or_float(vars)
        ind = np.cumsum([self.x.shape[0], self.numN])
        x, fN, fT = np.split(vars, ind)
        plant.multibody.SetPositionsAndVelocities(context, x)
        mu = plant.GetFrictionCoefficients(context)
        mu = np.diag(mu)
        # Match friction forces to normal forces
        return mu.dot(fN) - self._e.dot(fT)

    # Joint Limit Constraints
    def _joint_limit(self, vars):
        """
        Complementarity constraint between the position variables and the joint limit forces

        Arguments:
            Decision variable list:
                vars = state
        """
        plant, _, _ = self._autodiff_or_float(vars)
        # Get configuration and joint limit forces
        q = vars[0:plant.multibody.num_positions()]
        # Calculate distance from limits
        qmax = plant.multibody.GetPositionUpperLimits()
        qmin = plant.multibody.GetPositionLowerLimits()
        q_valid = np.isfinite(qmax)
        return np.concatenate((q[q_valid] - qmin[q_valid],
                                qmax[q_valid] - q[q_valid]),
                                axis=0)
     
    def _autodiff_or_float(self, z):
        """Returns the autodiff or float implementation of model and context based on the dtype of the decision variables"""
        if z.dtype == "float":
            return (self.plant_f, self.context_f, self.mbf_f)
        else:
            return (self.plant_ad, self.context_ad, self.mbf_ad)

    def _set_initial_timesteps(self):
        """Set the initial timesteps to their maximum values"""
        self.prog.SetInitialGuess(self.h, self.maximum_timestep*np.ones(self.h.shape))

    def set_initial_guess(self, xtraj=None, utraj=None, ltraj=None, jltraj=None):
        """Set the initial guess for the decision variables"""
        if xtraj is not None:
            self.prog.SetInitialGuess(self.x, xtraj)
        if utraj is not None:
            self.prog.SetInitialGuess(self.u, utraj)
        if ltraj is not None:
            self.prog.SetInitialGuess(self.l, ltraj)
        if jltraj is not None:
            self.prog.SetInitialGuess(self.jl, jltraj)

    def add_running_cost(self, cost_func, vars=None, name="RunningCost"):
        """Add a running cost to the program"""
        integrated_cost = lambda x: np.array(x[0] * cost_func(x[1:]))
        for n in range(0, self.num_time_samples-1):
            new_vars = [var[:,n] for var in vars]
            new_vars.insert(0, self.h[n,:])
            self.prog.AddCost(integrated_cost, np.concatenate(new_vars,axis=0), description=name)

    def add_tracking_cost(self, Q, traj, vars=None, name="TrackingCost"):
        """ 
        Add a quadratic running cost penalizing the difference from another trajectory
        
        Adds a running cost of the form:
            (z[k] - z0[k])^T Q (z[k] - z0[k])
        where z is the decision variable, z0 is the reference value, and Q is the positive-semidefinite weighting matrix
        
        Arguments:
            Q: an (N, N) numpy array of weights
            traj: an (N, M) array of reference values
            vars: a subset of the decision variables
            name (optional): a string describing the tracking cost
        """
        #TODO: Implement for tracking a trajectory / with variable timesteps
        for n in range(0, self.num_time_samples-1):
            integrated_cost = lambda z: z[0]*(z[:1] - traj[:,n]).dot(Q.dot(z[1:] - traj[:,n]))
            new_vars = [var[:,n] for var in vars]
            new_vars.insert(0, self.h[n,:])
            self.prog.AddCost(integrated_cost, np.concatenate(new_vars, axis=0), description=name)

    def add_final_cost(self, cost_func, vars=None, name="FinalCost"):
        """Add a final cost to the program"""
        if vars is not None:
            vars = np.concatenate(vars,axis=0)
            self.prog.AddCost(cost_func, vars, description=name)
        else:
            self.prog.AddCost(cost_func,description=name)
            
    def add_quadratic_running_cost(self, Q, b, vars=None, name="QuadraticCost"):
        """
        Add a quadratic running cost to the program
        
        Arguments:
            Q (numpy.array[n,n]): a square numpy array of cost weights
            b (numpy.array[n,1]): a vector of offset values
            vars (list): a list of program decision variables subject to the cost
            name (str, optional): a description of the cost function
        """
        integrated_cost = lambda z: z[0]*(z[1:]-b).dot(Q.dot(z[1:]-b))
        for n in range(0, self.num_time_samples-1):
            new_vars = [var[:,n] for var in vars]
            new_vars.insert(0, self.h[n,:])
            self.prog.AddCost(integrated_cost, np.concatenate(new_vars,axis=0), description=name)

    def add_equal_time_constraints(self):
        """impose that all timesteps be equal"""
       # Enforce the constraint with a linear constraint matrix of pairwise differences 
        num_h = self.h.shape[0]
        M = np.eye(num_h-1, num_h) - np.eye(num_h-1, num_h, 1)
        b = np.zeros((num_h-1,))
        self.prog.AddLinearEqualityConstraint(Aeq=M, beq=b, vars=self.h).evaluator().set_description('EqualTimeConstraints')
        
    def add_state_constraint(self, knotpoint, value, subset_index=None):
        """
        add a constraint to the state vector at a particular knotpoint
        
        Arguments:  
            knotpoint (int): the index of the knotpoint at which to add the constraint
            value (numpy.array): an array of constraint values
            subset_index: optional list of indices specifying which state variables are subject to constraint
        """
        if subset_index is None:
            subset_index = np.array(range(0, self.x.shape[0]))  
        # Check that the input is within the joint limits
        qmin = self.plant_f.multibody.GetPositionLowerLimits()
        qmax = self.plant_f.multibody.GetPositionUpperLimits()
        q_subset = subset_index[subset_index < self.plant_f.multibody.num_positions()]
        q = value[subset_index < self.plant_f.multibody.num_positions()]
        if any(q < qmin[q_subset]):
            raise ValueError("State constraint violates position lower limits")
        if any(q > qmax[q_subset]):
            raise ValueError("State constraint violates position upper limits")
        # Create the constraint
        A = np.eye(value.shape[0])   
        self.prog.AddLinearEqualityConstraint(Aeq=A, beq=value, vars=self.x[subset_index, knotpoint]).evaluator().set_description("StateConstraint")
            
    def add_control_limits(self, umin, umax):
        """
        adds acutation limit constraints to all knot pints
        
        Arguments:
            umin (numpy.array): array of minimum control effort limits
            umax (numpy.array): array of maximum control effort limits

        umin and umax must as many entries as there are actuators in the problem. If the control has no effort limit, use np.inf
        """
        #TODO check the inputs
        u_valid = np.isfinite(umin)
        for n in range(0, self.num_time_samples):
            self.prog.AddBoundingBoxConstraint(umin[u_valid], umax[u_valid], self.u[n, u_valid]).evaluator().set_description("ControlLimits")

    def initial_state(self):
        """returns the initial state vector"""
        return self.x[:,0]

    def final_state(self):
        """returns the final state vector"""
        return self.x[:,-1]

    def total_time(self):
        """returns the sum of the timesteps"""
        return sum(self.h)

    def get_program(self):
        """returns the stored mathematical program object for use with solve"""
        return self.prog

    def reconstruct_state_trajectory(self, soln):
        """Returns the state trajectory from the solution"""
        t = self.get_solution_times(soln)
        return PiecewisePolynomial.FirstOrderHold(t, soln.GetSolution(self.x))

    def reconstruct_input_trajectory(self, soln):
        """Returns the input trajectory from the solution"""
        t = self.get_solution_times(soln)
        return PiecewisePolynomial.FirstOrderHold(t, soln.GetSolution(self.u))
    
    def reconstruct_reaction_force_trajectory(self, soln):
        """Returns the reaction force trajectory from the solution"""
        t = self.get_solution_times(soln)
        return PiecewisePolynomial.FirstOrderHold(t, soln.GetSolution(self.l))
    
    def reconstruct_limit_force_trajectory(self, soln):
        """Returns the joint limit force trajectory from the solution"""
        if self.Jl is not None:
            t = self.get_solution_times(soln)
            return PiecewisePolynomial.FirstOrderHold(t, soln.GetSolution(self.jl))
        else:
            return None

    def reconstruct_all_trajectories(self, soln):
        """Returns state, input, reaction force, and joint limit force trajectories from the solution"""
        state = self.reconstruct_state_trajectory(soln)
        input = self.reconstruct_input_trajectory(soln)
        lforce = self.reconstruct_reaction_force_trajectory(soln)
        jlforce = self.reconstruct_limit_force_trajectory(soln)
        return (state, input, lforce, jlforce)

    def get_solution_times(self, soln):
        """Returns a vector of times for the knotpoints in the solution"""
        h = soln.GetSolution(self.h)
        t = np.concatenate((np.zeros(1,), h), axis=0)
        return np.cumsum(t)

    def result_to_dict(self, soln):
        """ unpack the trajectories from the program result and store in a dictionary"""
        t = self.get_solution_times(soln)
        x, u, f, jl = self.reconstruct_all_trajectories(soln)
        if jl is not None:
            jl = jl.vector_values(t)
        soln_dict = {"time": t,
                    "state": x.vector_values(t),
                    "control": u.vector_values(t), 
                    "force": f.vector_values(t),
                    "jointlimit": jl,
                    "solver": soln.get_solver_id().name(),
                    "success": soln.is_success(),
                    "exit_code": soln.get_solver_details().info,
                    "final_cost": soln.get_optimal_cost()
                    }
        return soln_dict

    def set_slack(self, val):
        self.distance_cstr.set_slack(val)
        self.sliding_cstr.set_slack(val)
        self.friccone_cstr.set_slack(val)

    def enable_cost_display(self, display='terminal'):
        """
        Add a visualization callback to print/show the cost values and constraint violations at each iteration

        Parameters:
            display: "terminal" prints the costs and constraints to the terminal
                     "figure" prints the costs and constraints to a figure window
                     "all"    prints the costs and constraints to the terminal and to a figure window
        """
        printer = MathProgIterationPrinter(prog=self.prog, display=display)
        all_vars = self.prog.decision_variables()
        self.prog.AddVisualizationCallback(printer, all_vars)

def integrate_quaternion(q, w, dt):
    """
    Integrate the unit quaternion q given it's associated angular velocity w

    Arguments:
        q: (4,) numpy array specifying a unit quaternion
        w: (3,) numpy array specifying an angular velocity
        dt: scalar indicating timestep 

    Return Values
        (4,) numpy array specifying the next step unit quaternion
    """
    # Initialize rotation quaternion
    Dq = np.zeros((4,),dtype=w.dtype)
    # Multiply velocity by time
    v = w * dt / 2.
    # Check for the case of v = 0
    if all(v == 0):
        Dq[0] = 1.
        Dq[1:] = v  #Still 0, but set to v for AutoDiff
    else:
    # normalize the velocity vector
        speed = np.linalg.norm(w)
        nw = w/speed
        # Calculate the rotation quaternion
        Dq[0] = np.cos(speed)
        Dq[1:] = nw * np.sin(speed)
    # Do the rotation for body-fixed angular rate
    return quaternion_product(q, Dq)

def quaternion_product(q1, q2):
    """
    Returns the quaternion product of two quaternions, q1*q2

    Arguments:
        q1: (4,) numpy array
        q2: (4,) numpy array
    """
    qprod = np.zeros((4,), dtype=type(q1))
    # Scalar part
    qprod[0] = q1[0]*q2[0] - np.dot(q1[1:], q2[1:])
    # Vector part
    qprod[1:] = q1[0]*q2[1:] + q2[0]*q1[1:] - np.cross(q1[1:], q2[1:])
    # Return
    return qprod


class NonlinearComplementarityFcn():
    """
    Implements a complementarity relationship involving a nonlinear function, such that:
        f(x) >= 0
        z >= 0
        f(x)*z <= s
    where f is the function, x and z are decision variables, and s is a slack parameter.
    By default s=0 (strict complementarity)
    """
    def __init__(self, fcn, xdim=0, zdim=1, slack=0.):
        self.fcn = fcn
        self.xdim = xdim
        self.zdim = zdim
        self.slack = slack
    
    def __call__(self, vars):
        """Evaluate the complementarity constraint """
        x, z = np.split(vars, [self.xdim])
        fcn_val = self.fcn(x)
        return np.concatenate((fcn_val, z, fcn_val * z - self.slack), axis=0)

    def lower_bound(self):
        return np.concatenate((np.zeros((2*self.zdim,)), -np.full((self.zdim,), np.inf)), axis=0)
    
    def upper_bound(self):
        return np.concatenate((np.full((2*self.zdim,), np.inf), np.zeros((self.zdim,))), axis=0)

    def set_slack(self, val):
        self.slack = val