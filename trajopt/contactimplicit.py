"""
contactimplicit: Implements Contact Implicit Trajectory Optimization using Backward Euler Integration
    Partially integrated with pyDrake
    contains pyDrake's MathematicalProgram to formulate and solve nonlinear programs
    uses pyDrake's MultibodyPlant to represent rigid body dynamics
Luke Drnach
October 5, 2020
"""

import numpy as np 
from matplotlib import pyplot as plt
from pydrake.all import MathematicalProgram, PiecewisePolynomial, Variable
from pydrake.autodiffutils import AutoDiffXd
from utilities import MathProgIterationPrinter, plot_complementarity
import trajopt.complementarity as compl
#TODO: Unit testing for whole-body and centrodial optimizers

class OptimizationOptions():
    """ Keeps track of optional settings for Contact Implicit Trajectory Optimization"""
    def __init__(self):
        """ Initialize the options to their default values"""
        self.__complementarity_class = compl.NonlinearComplementarityConstantSlack

    def useLinearComplementarityWithVariableSlack(self):
        """ Use linear complementarity with equality constraints"""
        self.__complementarity_class = compl.LinearEqualityVariableSlackComplementarity

    def useNonlinearComplementarityWithVariableSlack(self):
        """ Use nonlinear complementarity """
        self.__complementarity_class = compl.NonlinearComplementarityVariableSlack

    def useNonlinearComplementarityWithCost(self):
        """ Use nonlinear complementarity but enforce the equality constraint in a cost"""
        self.__complementarity_class = compl.CostRelaxedNonlinearComplementarity

    def useNonlinearComplementarityWithConstantSlack(self):
        """ Use a constant slack in the complementarity constraints"""
        self.__complementarity_class = compl.NonlinearComplementarityConstantSlack

    def useLinearComplementarityWithConstantSlack(self):
        """ Use a decision variable for the slack in the complementarity constraints"""
        self.__complementarity_class = compl.LinearEqualityConstantSlackComplementarity

    @property
    def complementarity(self):
        return self.__complementarity_class

class DecisionVariableList():
    """Helper class for adding a list of decision variables to a cost/constraint"""
    def __init__(self, varlist = []):
        self.var_list = varlist

    def add(self, new_vars):
        self.var_list.append(new_vars)

    def get(self, n):
        return np.concatenate([var[:,n] for var in self.var_list], axis=0)

class ContactImplicitDirectTranscription():
    """
    Implements contact-implicit trajectory optimization using Direct Transcription
    """
    def __init__(self, plant, context, num_time_samples, minimum_timestep, maximum_timestep, options=OptimizationOptions()):
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
        self.options = options
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
        # If there are floating variables, add the quaternion constraints
        if self._has_quaternion_states():
            self._add_quaternion_constraints()
        # Add contact constraints
        self._add_contact_constraints()
        # Initialize the timesteps
        self._set_initial_timesteps()

    def _check_floating_dof(self):
        # Get the floating bodies
        floating = self.plant_f.multibody.GetFloatingBaseBodies()
        self.floating_pos = []
        self.floating_vel = []
        self.floating_mag = []
        num_states = self.plant_f.multibody.num_positions() + self.plant_f.multibody.num_velocities()
        while len(floating) > 0:
            body = self.plant_f.multibody.get_body(floating.pop())
            if body.has_quaternion_dofs():
                self.floating_pos.append(body.floating_positions_start())
                self.floating_vel.append(body.floating_velocities_start())
                self.floating_mag.append(num_states)
                num_states += 1
    
    def _has_quaternion_states(self):
        return len(self.plant_f.multibody.GetFloatingBaseBodies()) > 0

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
<<<<<<< Updated upstream
        nX = 2*self.plant_ad.multibody.num_velocities()
        # if self._has_quaternion_states() and self.options.orientationType == OrientationType.QUATERNION:
        #     nQuat = len(self.floating_mag)
        #     self.x = self.prog.NewContinuousVariables(rows=nX+nQuat, cols=self.num_time_samples, name='x')
        # else:
        self.x = self.prog.NewContinuousVariables(rows=nX, cols=self.num_time_samples, name='x')
=======
        num_states = self.plant_ad.multibody.num_positions() + self.plant_ad.multibody.num_velocities()
        self.x = self.prog.NewContinuousVariables(rows = num_states, cols=self.num_time_samples, name='x')
>>>>>>> Stashed changes
        # Add control variables to the program
        nU = self.plant_ad.multibody.num_actuators()
        self.u = self.prog.NewContinuousVariables(rows=nU, cols=self.num_time_samples, name='u')
        # Add reaction force variables to the program
        numN = self.plant_f.num_contacts()
        numT = self.plant_f.num_friction()
        self._normal_forces = self.prog.NewContinuousVariables(rows = numN, cols=self.num_time_samples, name="normal_force")
        self._tangent_forces = self.prog.NewContinuousVariables(rows = numT, cols=self.num_time_samples, name="tanget_force")
        self._sliding_vel = self.prog.NewContinuousVariables(rows = numN, cols=self.num_time_samples, name="sliding_velocity")
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
        dynamics_bound = np.zeros((self.x.shape[0],1))
        # Check for joint limits first
        if self.Jl is not None:
            # Create the joint limit constraint
            self.joint_limit_cstr = compl.ConstantSlackNonlinearComplementarity(self._joint_limit, xdim=self.x.shape[0], zdim=self.jl.shape[0])
            for n in range(0, self.num_time_samples-1):
                # Add timestep constraints
                self.prog.AddBoundingBoxConstraint(self.minimum_timestep, self.maximum_timestep, self.h[n,:]).evaluator().set_description('TimestepConstraint')
                # Add dynamics constraints
                self.prog.AddConstraint(self._backward_dynamics, 
                            lb=dynamics_bound,
                            ub=dynamics_bound,
                            vars=np.concatenate((self.h[n,:], self.x[:,n], self.x[:,n+1], self.u[:,n], self._normal_forces[:,n+1], self._tangent_forces[:,n+1], self.jl[:,n+1]), axis=0),
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
                            lb=dynamics_bound,
                            ub=dynamics_bound,
                            vars=np.concatenate((self.h[n,:], self.x[:,n], self.x[:,n+1], self.u[:,n], self._normal_forces[:,n+1], self._tangent_forces[:,n+1]), axis=0),
                            description="dynamics")  
           
    def _add_contact_constraints(self):
        """ Add complementarity constraints for contact to the optimization problem"""
        # Create the constraint according to the implementation and slacktype options
        self.distance_cstr = self.options.complementarity(self._normal_distance, xdim=self.x.shape[0], zdim=self.numN)
        self.sliding_cstr = self.options.complementarity(self._sliding_velocity, xdim=self.x.shape[0]+self.numN, zdim = self.numT)
        self.friction_cstr = self.options.complementarity(self._friction_cone, xdim=self.x.shape[0]+self.numN + self.numT, zdim=self.numN)
        # Update the names
        self.distance_cstr.set_description("normal_distance")
        self.sliding_cstr.set_description("sliding_velocity")
        self.friction_cstr.set_description("friction_cone")
        # Add to program
        for n in range(self.num_time_samples):
            self.distance_cstr.addToProgram(self.prog, xvars=self.x[:,n], zvars=self._normal_forces[:,n])
            self.sliding_cstr.addToProgram(self.prog, xvars=np.concatenate([self.x[:,n],self._sliding_vel[:,n]], axis=0), zvars=self._tangent_forces[:,n])
            self.friction_cstr.addToProgram(self.prog, xvars=np.concatenate([self.x[:,n], self._normal_forces[:,n], self._tangent_forces[:,n]], axis=0), zvars=self._sliding_vel[:,n])

    def _backward_dynamics(self, z):  
        """
        backward_dynamics: Backward Euler integration of the dynamics constraints
        Decision variables are passed in through a list in the order:
            z = [h, x1, x2, u, l, jl]
        Returns the dynamics defect, evaluated using Backward Euler Integration. 
        """
        plant, context, mbf = self._autodiff_or_float(z)
        # Split the variables from the decision variables
        ind = np.cumsum([self.h.shape[1], self.x.shape[0], self.x.shape[0], self.u.shape[0], self._normal_forces.shape[0]])
        h, x1, x2, u, fN, fT = np.split(z, ind)
        # Calculate the position integration error
        p1, _ = np.split(x1, 2)
        p2, dp2 = np.split(x2, 2)
        fp = p2 - p1 - h*dp2
        # Get positions and velocities
        _, v1 = np.split(x1, [plant.multibody.num_positions()])
        q2, v2 = np.split(x2, [plant.multibody.num_positions()])
        # Update the context - backward Euler integration
        plant.multibody.SetPositionsAndVelocities(context, np.concatenate((q2,v2), axis=0))
        # Set mutlibodyForces to zero
        mbf.SetZero()
        # calculate generalized forces
        M = plant.multibody.CalcMassMatrixViaInverseDynamics(context)
        C = plant.multibody.CalcBiasTerm(context)
        G = plant.multibody.CalcGravityGeneralizedForces(context)
        B = plant.multibody.MakeActuationMatrix()
        # Integrated Generalized forces
        forces = (B.dot(u) - C + G)
        # Joint limits
        if self.Jl is not None:
            fT, jl = np.split(fT, [self._tangent_forces.shape[0]])
            forces += self.Jl.dot(jl)
        # Contact reaction forces
        Jn, Jt = plant.GetContactJacobians(context)
        forces += Jn.transpose().dot(fN) + Jt.transpose().dot(fT)
        # Do inverse dynamics - velocity dynamics error
        fv = M.dot(v2 - v1) - h*forces
        return np.concatenate((fp, fv), axis=0)

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
        q, v = np.split(state, plant.multibody.num_positions())
        plant.multibody.SetPositionsAndVelocities(context, np.concatenate((q,v), axis=0))    
        return plant.GetNormalDistances(context)

    def enforceNormalDissipation(self):
        """
        Add the constraints on the normal dissipation
        """
        for n in range(self.num_time_samples):
            self.prog.AddConstraint(self._normal_dissipation, 
                                lb = np.full((self.numN,), -np.inf), 
                                ub = np.zeros((self.numN,)),
                                vars=np.concatenate([self.x[:,n], self._normal_forces[:,n]], axis=0),
                                description="NormalDissipationConstraint")

    def _normal_dissipation(self, vars):
        """
        Condition on the normal force being dissipative
        """
        state, force = np.split(vars, [self.x.shape[0]])
<<<<<<< Updated upstream
        plant, context, _ = self._autodiff_or_float(state)
        q, v = self._dvars_to_coordinates(state)
=======
        plant, context = self._autodiff_or_float(state)
        q, v = np.split(state, plant.multibody.num_positions())
>>>>>>> Stashed changes
        plant.multibody.SetPositionsAndVelocities(context, np.concatenate((q,v), axis=0))
        Jn, _ = plant.GetContactJacobians(context)
        return Jn.dot(v) * force

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
        q, v = np.split(x, plant.multibody.num_positions())
        plant.multibody.SetPositionsAndVelocities(context, np.concatenate((q,v), axis=0))
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
        ind = np.cumsum([self.x.shape[0], self._normal_forces.shape[0]])
        x, fN, fT = np.split(vars, ind)
        q, v = np.split(x, plant.multibody.num_positions())
        plant.multibody.SetPositionsAndVelocities(context, np.concatenate((q,v), axis=0))
        mu = plant.GetFrictionCoefficients(context)
        mu = np.diag(mu)
        # Match friction forces to normal forces
        return mu.dot(fN) - self._e.dot(fT)

    # Joint Limit Constraints
    def _joint_limit(self, dvars):
        """
        Complementarity constraint between the position variables and the joint limit forces

        Arguments:
            Decision variable list:
                vars = state
        """
        plant, _, _ = self._autodiff_or_float(dvars)
        # Get configuration and joint limit forces
        q, _ = np.split(dvars, plant.multibody.num_positions())
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
        if type(subset_index) is not np.ndarray:
            subset_index = np.array(subset_index)
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

    def reconstruct_slack_trajectory(self, soln):
        
        # Get the slack variables from the complementarity problems
        if any(isinstance(slack, Variable) for slack in self.var_slack.flatten()):
            #Filter out 'Variable' types
            slack_vars = np.row_stack([slack_var for slack_var in self.var_slack if isinstance(slack_var[0], Variable)])
            return self.reconstruct_trajectory(slack_vars, soln)
        else:
            return None

    def reconstruct_all_trajectories(self, soln):
        """Returns state, input, reaction force, and joint limit force trajectories from the solution"""
        state = self.reconstruct_state_trajectory(soln)
        input = self.reconstruct_input_trajectory(soln)
        lforce = self.reconstruct_reaction_force_trajectory(soln)
        jlforce = self.reconstruct_limit_force_trajectory(soln)
        slacks = self.reconstruct_slack_trajectory(soln)
        return (state, input, lforce, jlforce, slacks)

    def get_solution_times(self, soln):
        """Returns a vector of times for the knotpoints in the solution"""
        h = soln.GetSolution(self.h)
        t = np.concatenate((np.zeros(1,), h), axis=0)
        return np.cumsum(t)

    def result_to_dict(self, soln):
        """ unpack the trajectories from the program result and store in a dictionary"""
        t = self.get_solution_times(soln)
        x, u, f, jl, s = self.reconstruct_all_trajectories(soln)
        if jl is not None:
            jl = jl.vector_values(t)
        if s is not None:
            s = s.vector_values(t)
        soln_dict = {"time": t,
                    "state": x.vector_values(t),
                    "control": u.vector_values(t), 
                    "force": f.vector_values(t),
                    "jointlimit": jl,
                    "slacks": s,
                    "solver": soln.get_solver_id().name(),
                    "success": soln.is_success(),
                    "exit_code": soln.get_solver_details().info,
                    "final_cost": soln.get_optimal_cost()
                    }
        return soln_dict
    
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

    def enable_iteration_visualizer(self):
        """
        Add a visualization callback to make a meshcat visualization after each iteration
        """
        self.prog.AddVisualizationCallback(self._visualize_iteration, self.prog.decision_variables())

    def _visualize_iteration(self, dvars):
        """
        Visualize the result using MeshCat
        """
        x = np.zeros(self.x.shape)
        for n in range(x.shape[0]):
            x[n,:] = dvars[self.prog.FindDecisionVariableIndices(self.x[n,:])]
        h = dvars[self.prog.FindDecisionVariableIndices(self.h)]
        t = np.cumsum(np.hstack([0.,h]))
        x = x[:, 0:self.plant_ad.multibody.num_positions() + self.plant_ad.multibody.num_velocities()]
        xtraj = PiecewisePolynomial.FirstOrderHold(t, x)
        self.plant_f.visualize(xtraj)

    def initialize_from_previous(self, result):
        """ Initialize the program from a previous solution to the same program """
        dvars = self.prog.decision_variables()
        dvals = result.GetSolution(dvars)
        self.prog.SetInitialGuess(dvars, dvals)

    @property
    def l(self):
        """Legacy property definition for backwards compatibility """
        return np.concatenate([self._normal_forces, self._tangent_forces, self._sliding_vel], axis=0)
    
    @property
    def numN(self):
        return self._normal_forces.shape[0]

    @property
    def numT(self):
        return self._tangent_forces.shape[0]

    @property
    def var_slack(self):
        """Variable slack for contact complementarity constraints"""
        return np.vstack([self.distance_cstr.slack, self.sliding_cstr.slack, self.friction_cstr.slack])
    
    @property
    def const_slack(self):
        """Constant slack for contact complementarity constraints"""
        return self.distance_cstr.slack

    @const_slack.setter
    def slack(self, val):
        """ Set the constant slack variable in the complementarity constraints """
        self.distance_cstr.slack = val
        self.sliding_cstr.slack = val
        self.friction_cstr.slack = val

    @property
    def complementarity_cost_weight(self):
        """Generic cost weight for complementarity constraints"""
        return self.distance_cstr.cost_weight

    @complementarity_cost_weight.setter
    def complementarity_cost_weight(self, val):
        """Set cost weight for complementarity cost """
        self.distance_cstr.cost_weight = val
        self.sliding_cstr.cost_weight = val
        self.friction_cstr.cost_weight = val

class CentroidalContactTranscription(ContactImplicitDirectTranscription):
    #TODO: Unit testing for all contact-implicit problems (Block, DoublePendulum, A1)

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
        num_states = self.plant_ad.num_velocities() + self.plant_ad.num_positions()
        self.x = self.prog.NewContinuousVariables(rows=num_states, cols=self.num_time_samples, name='x')
        # Add COM variables to the program
        self.com = self.prog.NewContinuousVariables(rows=6, cols=self.num_time_samples, name="com")
        # Add momentum variables
        self.momentum = self.prog.NewContinuousVariables(rows=3, cols=self.num_time_samples, name='momentum')
        # Add reaction force variables to the program
        numN = self.plant_f.num_contacts()
        numT = self.plant_ad.num_friction()
        self._normal_forces = self.prog.NewContinuousVariables(rows = numN, cols=self.num_time_samples, name='normal_forces')
        self._tangent_forces = self.prog.NewContinuousVariables(rows = numT, cols=self.num_time_samples, name='tangent_forces')
        self._sliding_vel = self.prog.NewContinuousVariables(rows = numN, cols = self.num_time_samples, name='sliding_vels')
        # Add contact points
        self.contactpts = self.prog.NewContinuousVariables(rows = 3*numN, cols=self.num_time_samples, name='contact_pts')
        # store a matrix for organizing the friction forces
        self._e = self.plant_ad.duplicator_matrix()
<<<<<<< Updated upstream
        # Add slack variables for complementarity problems
        # if self.options.ncc_implementation == NCCImplementation.LINEAR_EQUALITY: 
        #     if self.options.slacktype == NCCSlackType.VARIABLE_SLACK:
        #         self.slacks = self.prog.NewContinuousVariables(rows = 1 + 2*self.numN + self.numT, cols=self.num_time_sampels, name='slacks')
        #     else: 
        #         self.slacks = self.prog.NewContinuousVariables(rows=2*self.numN + self.numT, cols=self.num_time_samples, name='slacks')
        # elif self.options.ncc_implementation == NCCImplementation.NONLINEAR and self.options.slacktype == NCCSlackType.VARIABLE_SLACK:
        #     self.slacks = self.prog.NewContinuousVariables(rows=1,cols=self.num_time_samples, name='slacks')
        # else:
        #     self.slacks = []
=======
        
>>>>>>> Stashed changes

    def _add_dynamic_constraints(self):
        """
            Add in the centroidal dynamics and associated constraints to the problem
        """
        # Add equality constraints on the centrodial variables
        self._add_centroidal_constraints()
        # Add dynamics constraints between successive timepoints
        num_pos = self.plant_f.multibody.num_velocities()
        for n in range(0, self.num_time_samples-1):
                # Add timestep constraints
                self.prog.AddBoundingBoxConstraint(self.minimum_timestep, self.maximum_timestep, self.h[n,:]).evaluator().set_description('TimestepConstraint')
                # Add integrator constraints
                self.prog.AddConstraint(self._backward_com_dynamics, 
                            lb=np.zeros(shape=(self.com.shape[0], 1)),
                            ub=np.zeros(shape=(self.com.shape[0], 1)),
                            vars=np.concatenate((self.h[n,:], self.com[:,n], self.com[:,n+1], self._normal_forces[:,n+1], self._tangent_forces[:,n+1], self.contactpts[:,n+1]), axis=0),
                            description="com_dynamics")  
                self.prog.AddConstraint(self._backward_momentum_dynamics,
                            lb=np.zeros((3,1)),
                            ub=np.zeros((3,1)),
                            vars=np.concatenate([self.h[:,n], self.momentum[:,n], self.momentum[:,n+1],self.contactpts[:,n+1], self.com[0:3,n+1], self._normal_forces[:,n+1], self._tangent_forces[:,n+1]], axis=0),
                            description="momentum_dynamics")
                self.prog.AddConstraint(self._generalized_position_dynamics,
                            lb=np.zeros((num_pos,)),
                            ub=np.zeros((num_pos)),
                            vars=np.concatenate([self.h[:,n], self.x[0:num_pos,n], self.x[:,n+1]], axis=0),
                            description="position_dynamics")

    def _add_centroidal_constraints(self):
        """
        Add the constraints on the definitions of the centroidal variables
        """
        num_pos = self.plant_f.multibody.num_velocities()
        for n in range(self.num_time_samples):
            # Add equality constraints on centroidal variables and generalized coordinates
            self.prog.AddConstraint(self._com_error, 
                                    lb=np.zeros((3,1)),
                                    ub=np.zeros((3,1)),
                                    vars=np.concatenate([self.com[0:3,n], self.x[0:num_pos,n]]),
                                    description="com_error")
            self.prog.AddConstraint(self._momentum_error,
                                    lb=np.zeros((3,1)),
                                    ub=np.zeros((3,1)),
                                    vars=np.concatenate([self.momentum[:,n], self.x[:,n]]),
                                    description="momentum_error")
            self.prog.AddConstraint(self._contact_position_error, 
                                    lb=np.zeros((3*self.numN, 1)),
                                    ub=np.zeros((3*self.numN, 1)),
                                    vars=np.concatenate([self.contactpts[:,n], self.x[0:num_pos,n]], axis=0),
                                    description="contact_point_error")

    def _com_error(self, dvars):
        """
            Constraint on center of mass position 
        
            Decision variable list: [COM_POSITION, GEN_POSITIONS]
        """

        # Get the necessary plant
        plant, context, _ = self._autodiff_or_float(dvars)
        # Split the variables
        rCOM, q = np.split(dvars, [3])
        # Calculate COM position and return the constraint
        plant.multibody.SetPositions(context, q)
        return rCOM - plant.CalcCenterOfMassPositionInWorld(context)

    def _momentum_error(self, dvars):
        """
            Constraint on COM momentum variables
            Decision variable list: [COM_MOMENTUM, GEN_POSITIONS, GEN_VELOCITIES]
        """
        # Get the appropriate plant model
        plant, context, _ = self._autodiff_or_float(dvars)
        # Split the variables
        lCOM, state = np.split(dvars, [3])
        q, v = np.split(state, [plant.multibody.num_positions()])
        # Update the context
        plant.multibody.SetPositionsAndVelocities(context, q, v)
        # Return the momentum error
        pCOM = plant.CalcCenterOfMassPositionInWorld(context)
        return lCOM - plant.CalcSpatialMomentumInWorldAboutPoint(context, pCOM).rotational()

    def _contact_position_error(self, dvars):
        """
            Constraint for contact position variables

            Decision Variable List: [CONTACT_POINTS, GEN_POSITIONS]
        """
        # Get the plant model
        plant, context, _ = self._autodiff_or_float(dvars)
        # Split the variables
        contact, q = np.split(dvars, [3*self.numN])
        # Get the contact positions
        plant.multibody.SetPositions(context, q)
        kin = plant.calc_contact_positions(context)
        # Subtract the forward kinematic solution from the variables
        return contact - np.concatenate(kin, axis=0)

    def _backward_com_dynamics(self, dvars):
        """
            Returns the integration error for the COM Linear Dynamics

            Decision Variables: [timestep, CoM(k), CoM(k+1), Forces(k+1), ContactPoints(k+1)]
        """
        # Get the plant
        plant, _, _ = self._autodiff_or_float(dvars)
        # Split the variables
        h, rcom1, vcom1, rcom2, vcom2, forces, pts = np.split(dvars, np.cumsum([1, 3, 3, 3, 3, self.numN+self.numT]))
        # Position error (Midpoint Integration)
        p_err = rcom2 - rcom1 - h/2.0 * (vcom1 + vcom2)
        # Resolve forces in world coordinates
        forces = plant.resolve_contact_forces_in_world_at_points(forces, pts)
        netforce = np.zeros((3,), dtype=h.dtype)
        for force in forces:
            netforce += force
        # Velocity error (backward euler integration)
        v_err = vcom2 - vcom1 - h *(forces + self.total_mass() * plant.multibody.gravity_field())
        return np.concatenate((p_err, v_err), axis=0)

    def _backward_momentum_dynamics(self, dvars):
        """
            Returns the integration error for the COM Momentum Dynamics
        
            Decision Variables: [timestep, H(k), H(k+1), ContactPos(k), COM_POS(k), Forces(k)]
        """
        # Get the plant
        plant, _, _ = self._autodiff_or_float(dvars)
        # Split the variables
        h, momentum1, momentum2, contact_pos, com_pos, contact_force = np.split(dvars, np.cumsum([1, 3, 3, self.numN, 3]))
        # Resolve the contact force variables into world coordinates, then angularize
        forces = plant.resolve_contact_forces_in_world_at_points(contact_force, contact_pos)
        # Calculate cross products
        torque = np.zeros((3,), dtype = com_pos.dtype)
        for n in range(self.numN):
            torque += np.cross(contact_pos[3*n:3*(n+1)] - com_pos, forces[3*n:3*(n+1)])        
        # Calculate the defect
        return momentum2 - momentum1 - h * torque

    def _generalized_position_dynamics(self, dvars):
        """
            Returns the integration error for the generalized position dynamics

            Decision Variable List: [timestep, GEN_POS(k), GEN_POS(k+1), GEN_VEL(k+1)]
        """
        nV = self.plant_ad.multibody.num_velocities()
        h, q1, q2, v2 = np.split(dvars, np.cumsum([1, nV, nV]))
        return q2 - q1 - h*v2
        
    def _get_total_mass(self):
        mass = 0.
        body_inds = self.plant.GetBodyIndices(self.model_index)
        for ind in body_inds:
            mass += self.plant.get_body(ind).get_mass()
        return mass

    def _reconstruct_trajectory(self, soln, dvar):
        t = self.get_solution_times(soln)
        return PiecewisePolynomial.FirstOrderHold(t, soln.GetResult(dvar))

    def reconstruct_com_trajectory(self, soln):
        return self._reconstruct_trajectory(soln, self.com)
    
    def reconstruct_momentum_trajectory(self, soln):
        return self._reconstruct_trajectory(soln, self.momentum)

    def reconstruct_state_trajectory(self, soln):
        return self._reconstruct_trajectory(soln, self.x)

    def reconstruct_force_trajectory(self, soln):
        return self._reconstruct_trajectory(soln, self.l)

    def reconstruct_contact_point_trajectory(self, soln):
        return self._reconstruct_trajectory(soln, self.contactpts)

    def reconstruct_input_trajectory(self, soln):
        raise NotImplementedError("control torques not implemented in centroidal model")
    
    def reconstruct_limit_force_trajectory(self, soln):
        return NotImplementedError("joint limits not included in centroidal model")

    def reconstruct_all_trajectories(self, soln):
        com = self.reconstruct_com_trajectory(soln)
        momentum = self.reconstruct_momentum_trajectory(soln)
        state = self.reconstruct_state_trajectory(soln)
        contact = self.reconstruct_contact_point_trajectory(soln)
        forces = self.reconstruct_force_trajectory(soln)
        slacks = self.reconstruct_slack_trajectory(soln)
        return com, momentum, state, contact, forces, slacks

    def result_to_dict(self, soln):
        """ unpack the trajectories from the program result and store in a dictionary"""
        t = self.get_solution_times(soln)
        com, momentum, x, contactpts, forces, slacks = self.reconstruct_all_trajectories(soln)
        if slacks is not None:
            slacks = slacks.vector_values(t)
        return {"time": t,
                    "state": x.vector_values(t),
                    "com": com.vector_values(t), 
                    "force": forces.vector_values(t),
                    "momentum": momentum.vector_values(t),
                    "contactpts": contactpts.vector_values(t),
                    "slacks": slacks,
                    "solver": soln.get_solver_id().name(),
                    "success": soln.is_success(),
                    "exit_code": soln.get_solver_details().info,
                    "final_cost": soln.get_optimal_cost()
                    }

class ContactConstraintViewer():
    def __init__(self, trajopt, result_dict):
        self.trajopt = trajopt
        self._store_results(result_dict)

    def _store_results(self, result_dict):
        self.all_vals = np.zeros((self.trajopt.prog.num_vars(),))
        Finder = self.trajopt.prog.FindDecisionVariableIndices
        # Add in time variables
        self.all_vals[Finder(self.trajopt.h.flatten())] = np.diff(result_dict['time']).flatten()
        # Add in state variables
        self.all_vals[Finder(self.trajopt.x.flatten())] = result_dict['state'].flatten()
        # Add in control variables
        self.all_vals[Finder(self.trajopt.u.flatten())] = result_dict['control'].flatten()
        # Add in force variables
        self.all_vals[Finder(self.trajopt.l.flatten())] = result_dict['force'].flatten()
        # Add in limit variables (if any)
        if 'jointlimit' in result_dict and result_dict['jointlimit'] is not None:
            self.all_vals[Finder(self.trajopt.jl.flatten())] = result_dict['jointlimit'].flatten()
        # Add in slack variables (if any)
        if 'slacks' in result_dict and result_dict['slacks'] is not None:
            self.all_vals[Finder(self.trajopt.slacks.flatten())] = result_dict['slacks'].flatten()

    def plot_constraints(self):
        cstr_vals = self.calc_constraint_values()
        h_idx = self.trajopt.prog.FindDecisionVariableIndices(self.trajopt.h.flatten()) 
        time = np.cumsum(np.hstack((0, self.all_vals[h_idx])))
        self.plot_dynamic_defects(time, cstr_vals['dynamics'])
        if self.trajopt.options.complementarity in [compl.LinearEqualityConstantSlackComplementarity, compl.LinearEqualityVariableSlackComplementarity]:
<<<<<<< Updated upstream
        #if self.trajopt.options.ncc_implementation == NCCImplementation.LINEAR_EQUALITY:
=======
>>>>>>> Stashed changes
            self.plot_complementarity_linear(time, cstr_vals['normal_distance'], cstr_vals['sliding_velocity'], cstr_vals['friction_cone'])
        else:
            self.plot_complementarity_nonlinear(time, cstr_vals['normal_distance'], cstr_vals['sliding_velocity'], cstr_vals['friction_cone'])
        if len(self.trajopt.floating_pos) > 0:
            self.plot_quaternion_constraints(time, cstr_vals['unit_quaternion'], cstr_vals['unit_velocity_axis'], cstr_vals['quaternion_dynamics'])
        # Show the plot
        plt.show()

    def calc_constraint_values(self):
        """Return a dictionary of all constraint values"""
        # First get the values of the decision variables
        all_cstr = {}
        for cstr in self.trajopt.prog.GetAllConstraints():
            # Get the variables for the constraint
            dvars = cstr.variables()
            dvals = self.all_vals[self.trajopt.prog.FindDecisionVariableIndices(dvars)]
            # Evaluate the constraint
            cval = cstr.evaluator().Eval(dvals)
            # Sort by constraint name
            cname = cstr.evaluator().get_description()
            if cname in all_cstr.keys():
                all_cstr[cname].append(cval)
            else:
                all_cstr[cname] = [cval]
        # Convert each of the constraint violations into a numpy array
        for key in all_cstr.keys():
            all_cstr[key] = np.vstack(all_cstr[key]).transpose()
        return all_cstr

    def plot_quaternion_constraints(self, time, unit_pos, unit_vel, dynamics):
        fig, axs = plt.subplots(3,1)
        axs[0].plot(time, unit_pos.transpose(), linewidth=1.5)
        axs[0].set_ylabel("Unit quaternion")
        axs[1].plot(time, unit_vel.transpose(), linewidth=1.5)
        axs[1].set_ylabel("Unit velocity axis")
        axs[2].plot(time[1:], dynamics.transpose(), linewidth=1.5)
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Quaterion Dynamics")

    def plot_dynamic_defects(self, time, cstr_vals):
        fig, axs = plt.subplots(2,1)
        num_pos = self.trajopt.plant_f.multibody.num_positions() - 4*len(self.trajopt.floating_pos)
        fq = cstr_vals[0:num_pos,:]
        fv = cstr_vals[num_pos:,:]
        axs[0].plot(time[1:], fq.transpose(), linewidth=1.5)
        axs[0].set_ylabel('Position')
        axs[1].plot(time[1:], fv.transpose(), linewidth=1.5)
        axs[1].set_ylabel('Velocity')
        axs[1].set_xlabel('Time (s)')
        axs[0].set_title('Dynamics Constraint Defects')

    def plot_complementarity_linear(self, time, distance, velocity, friction):
        # Split the slack variable defects from the results
        Nslack, distance = np.split(distance, [self.trajopt.numN])
        Tslack, velocity = np.split(velocity, [self.trajopt.numT])
        Fslack, friction = np.split(friction, [self.trajopt.numN])
        # Plot the remaining variables using the nonlinear plotter
        normal_axs, tangent_axs = self.plot_complementarity_nonlinear(time, distance, velocity, friction)
        # Add in the slack variables
        color = 'tab:green'
        for n in range(0, self.trajopt.numN):
            normal_axs[2*n].plot(time, Nslack[n,:], linewidth=1.5, color=color)
            normal_axs[2*n+1].plot(time, Fslack[n,:], linewidth=1.5, color=color)
            for k in range(0, 4*self.trajopt.plant_f.dlevel):
                tangent_axs[n*4*self.trajopt.plant_f.dlevel + k].plot(time, Tslack[n*4*self.trajopt.plant_f.dlevel + k,:], linewidth=1.5, color=color)
        
    def plot_complementarity_nonlinear(self, time, distance, velocity, friction):
        # Get the variables
        norm_dist = distance[0:self.trajopt.numN,:]
        norm_force = distance[self.trajopt.numN:2*self.trajopt.numN,:]
        slide_vel = velocity[0:self.trajopt.numT,:]
        fric_force = velocity[self.trajopt.numT:2*self.trajopt.numT,:]
        fric_cone = friction[0:self.trajopt.numN,:]
        vel_slack = friction[self.trajopt.numN:2*self.trajopt.numN,:]
        # Plot the complementarity variables
        # Total number of plots per figure
        normal_axs = []
        tangent_axs = []
        for n in range(0,self.trajopt.numN):            
            _, naxs = plt.subplots(2,1)
            _, taxs = plt.subplots(4*self.trajopt.plant_f.dlevel,1)
            # Normal distance complementarity
            plot_complementarity(naxs[0], time,norm_dist[n,:], norm_force[n,:], 'distance','normal force')
            # Friction force complementarity
            for k in range(0, 4*self.trajopt.plant_f.dlevel):
                plot_complementarity(taxs[k], time, fric_force[n*4*self.trajopt.plant_f.dlevel + k,:], slide_vel[n*4*self.trajopt.plant_f.dlevel + k,:], 'Friction force','Tangent velocity')
            taxs[0].set_title(f"Contact point {n}")
            taxs[-1].set_xlabel('Time (s)')
            # Friction cone complementarity
            plot_complementarity(naxs[1], time,fric_cone[n,:], vel_slack[n,:], 'friction cone','sliding velocity')
            naxs[1].set_xlabel('Time (s)')
            naxs[0].set_title(f"Contact point {n}")
            #Collect the axes
            normal_axs.append(naxs)
            tangent_axs.append(taxs)
        return np.concatenate(normal_axs, axis=0), np.concatenate(tangent_axs, axis=0)

