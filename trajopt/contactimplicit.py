"""
contactimplicit: Implements Contact Implicit Trajectory Optimization using Backward Euler Integration
    Partially integrated with pyDrake
    contains pyDrake's MathematicalProgram to formulate and solve nonlinear programs
    uses pyDrake's MultibodyPlant to represent rigid body dynamics
Luke Drnach
October 5, 2020
"""

import numpy as np 
import timeit
from datetime import date
from matplotlib import pyplot as plt
from pydrake.all import MathematicalProgram, PiecewisePolynomial, Variable, SnoptSolver, IpoptSolver
from pydrake.autodiffutils import AutoDiffXd
import utilities as utils
import trajopt.complementarity as compl
import decorators as deco
#TODO: Unit testing for whole-body and centrodial optimizers

class OptimizationOptions():
    """ Keeps track of optional settings for Contact Implicit Trajectory Optimization"""
    def __init__(self):
        """ Initialize the options to their default values"""
        self.__complementarity_class = compl.NonlinearConstantSlackComplementarity

    def useLinearComplementarityWithVariableSlack(self):
        """ Use linear complementarity with equality constraints"""
        self.__complementarity_class = compl.LinearEqualityVariableSlackComplementarity

    def useNonlinearComplementarityWithVariableSlack(self):
        """ Use nonlinear complementarity """
        self.__complementarity_class = compl.NonlinearVariableSlackComplementarity

    def useNonlinearComplementarityWithCost(self):
        """ Use nonlinear complementarity but enforce the equality constraint in a cost"""
        self.__complementarity_class = compl.CostRelaxedNonlinearComplementarity

    def useNonlinearComplementarityWithConstantSlack(self):
        """ Use a constant slack in the complementarity constraints"""
        self.__complementarity_class = compl.NonlinearConstantSlackComplementarity

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

class OptimizationBase():
    def __init__(self):
        self.prog = MathematicalProgram()
        self.solver = SnoptSolver()
        self.solveroptions = {}
        self._elapsed = 0

    def useSnoptSolver(self):
        self.solver = SnoptSolver()

    def useIpoptSolver(self):
        self.solver = IpoptSolver()

    def setSolverOptions(self, options_dict={}):
        for key in options_dict:
            self.prog.SetSolverOption(self.solver.solver_id(), key, options_dict[key])
            self.solveroptions[key] = options_dict[key]

    def solve(self):
        
        print("Solving optimization:")
        start = timeit.default_timer()
        result = self.solver.Solve(self.prog)
        stop = timeit.default_timer()
        self._elapsed = stop - start
        print(f"Elapsed time: {self._elapsed}")
        return result

    def generate_report(self, result=None):
        """Generate a solution report from the solver"""
        text = f"Solver: {type(self.solver).__name__}\n"
        hrs, rem = divmod(self._elapsed, 3600)
        min, sec = divmod(rem, 60)
        text += f"Solver halted after {hrs:.0f} hours, {min:.0f} minutes, and {sec:.2f} seconds\n"
        if result is not None:
            text += utils.printProgramReport(result, self.prog, terminal=False, filename=None, verbose=True)
        text += f"Solver options:\n"
        if self.solveroptions is not {}:
            for key in self.solveroptions:
                text += f"\t {key}: {self.solveroptions[key]}\n"
        return text

class ContactImplicitDirectTranscription(OptimizationBase):
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
        super(ContactImplicitDirectTranscription, self).__init__()
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
        self.printer=None
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
        # Create a string for recording the optimization parameters
        self._text = {"header": f"{type(self).__name__} with {self.plant_f.str()}\n",
        "StateConstraints": '',
        "RunningCosts": '',
        "FinalCosts": '',
        "NormalDissipation": 'False',
        "EqualTime": 'False'}
        self._text['header'] += f"\tKnot points: {num_time_samples}\n\tTime range: [{(num_time_samples-1)*minimum_timestep},{(num_time_samples-1)*maximum_timestep}]\n\t"
        # Set force and control scaling variables
        self.force_scaling = 1.
        self.control_scaling = 1.

    def generate_report(self, result=None):
        # Generate a report string. Start with the header
        report = self._text['header']
        # Add in the date
        report += f"\nDate: {date.today().strftime('%B %d, %Y')}\n"
        # Add the total number of variables, the number of costs, and the number of constraints
        report += f"\nProblem has {self.prog.num_vars()} variables, {len(self.prog.GetAllCosts())} cost terms, and {len(self.prog.GetAllConstraints())} constraints\n\n"
        # Next add the report strings of the complementarity constraints
        report += self.distance_cstr.str() + "\n"
        report += self.sliding_cstr.str() + "\n"
        report += self.friction_cstr.str() + "\n"
        report += f"\nNormal Dissipation Enforced? {self._text['NormalDissipation']}"
        report += f"\nEqual time steps enforced? {self._text['EqualTime']}\n"
        # Add state constraints
        report += f"\nState Constraints: {self._text['StateConstraints']}\n"
        report += f"\nRunning Costs: {self._text['RunningCosts']}\n"
        report += f"\nFinal Costs: {self._text['FinalCosts']}\n"
        # Concatenate the report from the solver
        report += super(ContactImplicitDirectTranscription, self).generate_report(result)
        # Add the number of iterations, if possible
        if self.printer is not None:
            report += f"\nSolver halted after {self.printer.iteration} iterations\n"
        return report

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
        num_states = self.plant_ad.multibody.num_positions() + self.plant_ad.multibody.num_velocities()
        self.x = self.prog.NewContinuousVariables(rows = num_states, cols=self.num_time_samples, name='x')
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
            self.joint_limit_cstr = compl.NonlinearConstantSlackComplementarity(self._joint_limit, xdim=self.x.shape[0], zdim=self.jl.shape[0])
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
        plant, context = self._autodiff_or_float(z)
        # Split the variables from the decision variables
        ind = np.cumsum([self.h.shape[1], self.x.shape[0], self.x.shape[0], self.u.shape[0], self._normal_forces.shape[0]])
        h, x1, x2, u, fN, fT = np.split(z, ind)
        # Rescale force and control variables
        u = self.control_scaling * u
        fN = self.force_scaling * fN
        fT = self.force_scaling * fT
        # Calculate the position integration error
        p1, _ = np.split(x1, 2)
        p2, dp2 = np.split(x2, 2)
        fp = p2 - p1 - h*dp2
        # Get positions and velocities
        _, v1 = np.split(x1, [plant.multibody.num_positions()])
        q2, v2 = np.split(x2, [plant.multibody.num_positions()])
        # Update the context - backward Euler integration
        plant.multibody.SetPositionsAndVelocities(context, np.concatenate((q2,v2), axis=0))
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
        plant, context = self._autodiff_or_float(state)
        # Calculate the normal distance
        q, v = np.split(state, [plant.multibody.num_positions()])
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
        self._text['NormalDissipation'] = 'True'

    def _normal_dissipation(self, vars):
        """
        Condition on the normal force being dissipative
        """
        state, force = np.split(vars, [self.x.shape[0]])
        plant, context = self._autodiff_or_float(state)
        q, v = np.split(state, [plant.multibody.num_positions()])
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
        plant, context= self._autodiff_or_float(vars)
        # Split variables from the decision list
        x, gam = np.split(vars, [self.x.shape[0]])
        # Get the velocity, and convert to qdot
        q, v = np.split(x, [plant.multibody.num_positions()])
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
        plant, context = self._autodiff_or_float(vars)
        ind = np.cumsum([self.x.shape[0], self._normal_forces.shape[0]])
        x, fN, fT = np.split(vars, ind)
        q, v = np.split(x, [plant.multibody.num_positions()])
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
        plant, _ = self._autodiff_or_float(dvars)
        # Get configuration and joint limit forces
        q, _ = np.split(dvars, [plant.multibody.num_positions()])
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
            return (self.plant_f, self.context_f)
        else:
            return (self.plant_ad, self.context_ad)

    def _set_initial_timesteps(self):
        """Set the initial timesteps to their maximum values"""
        self.prog.SetInitialGuess(self.h, self.maximum_timestep*np.ones(self.h.shape))

    def set_initial_guess(self, xtraj=None, utraj=None, ltraj=None, jltraj=None, straj=None):
        """Set the initial guess for the decision variables"""
        if xtraj is not None:
            self.prog.SetInitialGuess(self.x, xtraj)
        if utraj is not None:
            self.prog.SetInitialGuess(self.u, utraj/self.control_scaling)
        if ltraj is not None:
            ltraj[:self.numN + self.numT,:] = ltraj[:self.numN + self.numT, :]/self.force_scaling
            self.prog.SetInitialGuess(self.l, ltraj)
        if jltraj is not None:
            self.prog.SetInitialGuess(self.jl, jltraj)
        if straj is not None:
            self.prog.SetInitialGuess(self.var_slack, straj)

    def add_running_cost(self, cost_func, vars=None, name="RunningCost"):
        """Add a running cost to the program"""
        integrated_cost = lambda x: np.array(x[0] * cost_func(x[1:]))
        for n in range(0, self.num_time_samples-1):
            new_vars = [var[:,n] for var in vars]
            new_vars.insert(0, self.h[n,:])
            self.prog.AddCost(integrated_cost, np.concatenate(new_vars,axis=0), description=name)
        # Add string representing the cost
        varnames = ', '.join([var.item(0).get_name().split('(')[0] for var in vars])
        self._text['RunningCosts'] += f"\n\t{name}: {cost_func.__name__} on {varnames}"

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
        # Add string representing the cost
        varnames = ', '.join([var.item(0).get_name().split('(')[0] for var in vars])
        self._text['FinalCosts'] += f"\n\t{name}: {cost_func.__name__} on {varnames}"
            
    def add_quadratic_running_cost(self, Q, b, vars=None, name="QuadraticCost"):
        """
        Add a quadratic running cost to the program
        
        Arguments:
            Q (numpy.array[n,n]): a square numpy array of cost weights
            b (numpy.array[n,1]): a vector of offset values
            vars (list): a list of program decision variables subject to the cost
            name (str, optional): a description of the cost function
        """
        # Input checking
        if vars is None:
            return
        if type(vars) != list:
            vars = [vars]
        # Add the cost
        integrated_cost = lambda z: z[0]*(z[1:]-b).dot(Q.dot(z[1:]-b))
        for n in range(0, self.num_time_samples-1):
            new_vars = [var[:,n] for var in vars]
            new_vars.insert(0, self.h[n,:])
            self.prog.AddCost(integrated_cost, np.concatenate(new_vars,axis=0), description=name)
        # Add string representing the cost
        varnames = ", ".join([var.item(0).get_name().split('(')[0] for var in vars])
        self._text['RunningCosts'] += f"\n\t{name}: Quadratic cost on {varnames} with weights Q = \n{Q} \n\tand bias b = \n{b}"

    def add_equal_time_constraints(self):
        """impose that all timesteps be equal"""
       # Enforce the constraint with a linear constraint matrix of pairwise differences 
        num_h = self.h.shape[0]
        M = np.eye(num_h-1, num_h) - np.eye(num_h-1, num_h, 1)
        b = np.zeros((num_h-1,))
        self.prog.AddLinearEqualityConstraint(Aeq=M, beq=b, vars=self.h).evaluator().set_description('EqualTimeConstraints')
        self._text['EqualTime'] = 'True'

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
        # Add string representing the state constraint
        self._text['StateConstraints'] += f"\n\tx[{knotpoint}, {subset_index}] = {value}"

    def add_control_limits(self, umin, umax):
        """
        adds acutation limit constraints to all knot pints
        
        Arguments:
            umin (numpy.array): array of minimum control effort limits
            umax (numpy.array): array of maximum control effort limits

        umin and umax must as many entries as there are actuators in the problem. If the control has no effort limit, use np.inf
        """
        #TODO check the inputs
        #TODO: Check for control scaling
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
        return PiecewisePolynomial.FirstOrderHold(t, self.control_scaling * soln.GetSolution(self.u))
    
    def reconstruct_reaction_force_trajectory(self, soln):
        """Returns the reaction force trajectory from the solution"""
        t = self.get_solution_times(soln)
        fN = self.force_scaling * soln.GetSolution(self._normal_forces)
        fT = self.force_scaling * soln.GetSolution(self._tangent_forces)
        gam = soln.GetSolution(self._sliding_vel)
        l = np.concatenate([fN, fT, gam], axis=0)
        return PiecewisePolynomial.FirstOrderHold(t, l)
    
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
            t = self.get_solution_times(soln)
            return PiecewisePolynomial.FirstOrderHold(t, soln.GetSolution(self.var_slack))
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
                    "final_cost": soln.get_optimal_cost(),
                    "duals": utils.getDualSolutionDict(self.prog, soln)
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
        self.printer = utils.MathProgIterationPrinter(prog=self.prog, display=display)
        all_vars = self.prog.decision_variables()
        self.prog.AddVisualizationCallback(self.printer, all_vars)

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
        # Initialize the primal variables
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
        return np.row_stack([self.distance_cstr.slack, self.sliding_cstr.slack, self.friction_cstr.slack])
    
    @property
    def const_slack(self):
        """Constant slack for contact complementarity constraints"""
        return self.distance_cstr.slack

    @const_slack.setter
    def const_slack(self, val):
        """ Set the constant slack variable in the complementarity constraints """
        self.distance_cstr.slack = val
        self.sliding_cstr.slack = val
        self.friction_cstr.slack = val

    @property
    def complementarity_cost_weight(self):
        """Generic cost weight for complementarity constraints
        
        Returns a tuple containing the cost weights for the normal distance, sliding velocity, and friction cone constraints, in that order
        """
        return (self.distance_cstr.cost_weight, self.sliding_cstr.cost_weight, self.friction_cstr.cost_weight)

    @complementarity_cost_weight.setter
    def complementarity_cost_weight(self, val):
        """Set cost weight for complementarity cost """
        if type(val) == float or type(val) == int:
            val = (val, val, val)
        elif len(val) != 3:
            raise ValueError("Complementarity cost weight must be either a nonnegative scalar or a triple of nonnegative scalars")
        self.distance_cstr.cost_weight = val[0]
        self.sliding_cstr.cost_weight = val[1]
        self.friction_cstr.cost_weight = val[2]

    @property
    def force_scaling(self):
        return self.__force_scaling

    @force_scaling.setter
    def force_scaling(self, val):
        if (isinstance(val, int) or isinstance(val, float)) and val > 0:
            self.__force_scaling = val
        else:
            raise ValueError(f"force_scaling must be a positive numeric value") 

    @property
    def control_scaling(self):
        return self.__control_scaling

    @control_scaling.setter
    def control_scaling(self, val):
        if (isinstance(val, int) or isinstance(val, float)) and val > 0:
            self.__control_scaling = val
        else:
            raise ValueError("control_scaling must be a positive numeric value")

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
        plant, context = self._autodiff_or_float(dvars)
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
        plant, context = self._autodiff_or_float(dvars)
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
        plant, context = self._autodiff_or_float(dvars)
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
        plant, _ = self._autodiff_or_float(dvars)
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
        plant, _ = self._autodiff_or_float(dvars)
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
        # Store the dual variables
        if 'duals' in result_dict:
            self.duals = result_dict['duals']

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

    @deco.showable_fig
    def plot_constraints(self, show_duals=True, savename=None):
        """Plot the dynamics, complementarity constraints, and dual solutions from trajectory optimization"""
        # Calculate the constraint values
        cstr_vals = self.calc_constraint_values()
        h_idx = self.trajopt.prog.FindDecisionVariableIndices(self.trajopt.h.flatten()) 
        time = np.cumsum(np.hstack((0, self.all_vals[h_idx])))
        all_figs = []
        all_axs = []
        # Plot the dynamics defects
        dyn_fig, dyn_axs = self.plot_dynamic_defects(time, cstr_vals['dynamics'], show=False, savename=utils.append_filename(savename, 'dynamics'))
        all_figs.append(dyn_fig)
        all_axs.append(dyn_axs)
        # Plot the complementarity constraints
        if self.trajopt.options.complementarity in [compl.LinearEqualityConstantSlackComplementarity, compl.LinearEqualityVariableSlackComplementarity]:
            cpl_figs, cpl_naxs, cpl_taxs = self.plot_complementarity_linear(time, cstr_vals['normal_distance'], cstr_vals['sliding_velocity'], cstr_vals['friction_cone'], show=False, savename=utils.append_filename(savename,'complementarity'))
        else:
            cpl_figs, cpl_naxs, cpl_taxs = self.plot_complementarity_nonlinear(time, cstr_vals['normal_distance'], cstr_vals['sliding_velocity'], cstr_vals['friction_cone'], show=False, savename=utils.append_filename(savename, 'complementarity'))
        all_figs.extend(cpl_figs)
        all_axs.append(cpl_naxs)
        all_axs.append(cpl_taxs)
        #Plot quaternion constraints, if any
        if len(self.trajopt.floating_pos) > 0:
            quat_fig, quat_axs = self.plot_quaternion_constraints(time, cstr_vals['unit_quaternion'], cstr_vals['unit_velocity_axis'], cstr_vals['quaternion_dynamics'], show=False, savename=utils.append_filename(savename, 'quaternions'))
            all_figs.append(quat_fig)
            all_axs.append(quat_axs)
        #Plot dual solutions, if desired
        if show_duals:
            dual_fig, dual_axs = self.plot_duals(time, show=False, savename=utils.append_filename(savename, "duals"))
            all_figs.extend(dual_fig)
            all_axs.extend(dual_axs)
        return all_figs, all_axs

    @deco.showable_fig
    def plot_duals(self, time, savename=None):
        """Plot the dual solutions """
        dual_figs = []
        dual_axs = []
        # Transpose all results in dual solution
        for key in self.duals:
            self.duals[key] = self.duals[key].transpose()
        # Plot the dual solutions and update figure labels
        dyn_fig, dyn_axs = self.plot_dynamic_defects(time, self.duals['dynamics'], show=False, savename=utils.append_filename(savename, 'dynamics'))
        dyn_axs[0].set_title('Dynamics Dual Variables')
        dual_figs.append(dyn_fig)
        dual_axs.append(dyn_axs) 
        if self.trajopt.options.complementarity in [compl.LinearEqualityConstantSlackComplementarity, compl.LinearEqualityVariableSlackComplementarity]:
            cfigs, naxs, taxs = self.plot_complementarity_linear(time, self.duals['normal_distance'], self.duals['sliding_velocity'], self.duals['friction_cone'], show=False, savename=utils.append_filename(savename, 'complementarity'))
        else:
            cfigs, naxs, taxs = self.plot_complementarity_nonlinear(time, self.duals['normal_distance'], self.duals['sliding_velocity'], self.duals['friction_cone'], show=False, savename=utils.append_filename(savename, 'complementarity'))
        for n in range(self.trajopt.numN):
            naxs[2*n].set_title(f"Contact point {n} Dual Variables")
            taxs[n*4*self.trajopt.plant_f.dlevel].set_title(f"Contact point {n} Dual Variables")
        dual_figs.extend(cfigs)
        dual_axs.append(naxs)
        dual_axs.append(taxs)
        # Check for quaternion constraints
        if len(self.trajopt.floating_pos) > 0:
            q_fig, q_axs = self.plot_quaternion_constraints(time, self.duals['unit_quaternion'], self.duals['unit_velocity_axis'],self.duals['quaternion_dynamics'], show=False, savename=utils.append_filename(savename, 'quaternion'))
            q_axs[0].set_title('Dual Variables')
            dual_figs.append(q_fig)
            dual_axs.append(q_axs)
        return dual_figs, dual_axs

    @deco.saveable_fig
    @deco.showable_fig
    def plot_quaternion_constraints(self, time, unit_pos, unit_vel, dynamics):
        """Plot the quaternion constriants"""
        fig, axs = plt.subplots(3,1)
        axs[0].plot(time, unit_pos.transpose(), linewidth=1.5)
        axs[0].set_ylabel("Unit quaternion")
        axs[1].plot(time, unit_vel.transpose(), linewidth=1.5)
        axs[1].set_ylabel("Unit velocity axis")
        axs[2].plot(time[1:], dynamics.transpose(), linewidth=1.5)
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Quaterion Dynamics")
        return fig, axs

    @deco.saveable_fig
    @deco.showable_fig
    def plot_dynamic_defects(self, time, cstr_vals):
        """ Plot the dynamic residuals"""
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
        return fig, axs

    @deco.saveable_fig
    @deco.showable_fig
    def plot_complementarity_linear(self, time, distance, velocity, friction):
        # Split the slack variable defects from the results
        Nslack, distance = np.split(distance, [self.trajopt.numN])
        Tslack, velocity = np.split(velocity, [self.trajopt.numT])
        Fslack, friction = np.split(friction, [self.trajopt.numN])
        # Plot the remaining variables using the nonlinear plotter
        figs, normal_axs, tangent_axs = self.plot_complementarity_nonlinear(time, distance, velocity, friction)
        # Add in the slack variables
        color = 'tab:green'
        for n in range(0, self.trajopt.numN):
            normal_axs[2*n].plot(time, Nslack[n,:], linewidth=1.5, color=color)
            normal_axs[2*n+1].plot(time, Fslack[n,:], linewidth=1.5, color=color)
            for k in range(0, 4*self.trajopt.plant_f.dlevel):
                tangent_axs[n*4*self.trajopt.plant_f.dlevel + k].plot(time, Tslack[n*4*self.trajopt.plant_f.dlevel + k,:], linewidth=1.5, color=color)
        return figs, normal_axs, tangent_axs

    @deco.saveable_fig
    @deco.showable_fig
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
        figs = []
        normal_axs = []
        tangent_axs = []
        for n in range(0,self.trajopt.numN):            
            nfig, naxs = plt.subplots(2,1)
            tfig, taxs = plt.subplots(4*self.trajopt.plant_f.dlevel,1)
            # Normal distance complementarity
            utils.plot_complementarity(naxs[0], time,norm_dist[n,:], norm_force[n,:], 'distance','normal force')
            # Friction force complementarity
            for k in range(0, 4*self.trajopt.plant_f.dlevel):
                utils.plot_complementarity(taxs[k], time, fric_force[n*4*self.trajopt.plant_f.dlevel + k,:], slide_vel[n*4*self.trajopt.plant_f.dlevel + k,:], 'Friction force','Tangent velocity')
            taxs[0].set_title(f"Contact point {n}")
            taxs[-1].set_xlabel('Time (s)')
            # Friction cone complementarity
            utils.plot_complementarity(naxs[1], time,fric_cone[n,:], vel_slack[n,:], 'friction cone','sliding velocity')
            naxs[1].set_xlabel('Time (s)')
            naxs[0].set_title(f"Contact point {n}")
            #Collect the axes
            normal_axs.append(naxs)
            tangent_axs.append(taxs)
            # Collect the figures
            figs.append(nfig)
            figs.append(tfig)
        return figs, np.concatenate(normal_axs, axis=0), np.concatenate(tangent_axs, axis=0)
