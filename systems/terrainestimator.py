"""
terrainestimator: Terrain estimation algorithms

Luke Drnach
November 10, 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import MathematicalProgram, MultibodyForces, Solve
from utilities import plot_complementarity
#TODO: Write GaussianProcessTerrain with Update methods. Test on actual data
#TODO: Implement generic terrain estimation
#TODO: Implement terrain estimation subclass with forces given
#TODO: Implement terrain estimation subclass for single parameter
#TODO: Implement terrain estimation subclass for pointwise estimates
#TODO: Implement terrain estimation subclass for orientation

class TerrainEstimator():
    """
    Base class for estimating terrain parameters

    Terrain estimator assumes the complete solution to the dynamics is given (the kinematics and kinetics)
    Kinetics / reaction forces must be given in terms of positive-only coefficients
    TerrainEstimator solves for the errors in the friction coefficient and normal distance, given the forces
    """
    def __init__(self, timesteppingplant=None):
        """ 
        Create a TerrainEstimator. 
        Initialization stores a reference to the plant and creates a Mathematical Program
        """
        self.plant = timesteppingplant
        self.mbf = MultibodyForces(self.plant.multibody)
        self.context = self.plant.multibody.CreateDefaultContext()
        self.prog = MathematicalProgram()
        self._create_decision_variables()

    def _create_decision_variables(self):
        """Create the decision variables for the terrain estimation program """
        # Assume linear error in distance and friction 
        self.dist_err = self.prog.NewContinuousVariables(self.plant.num_contacts())
        self.fric_err = self.prog.NewContinuousVariables(self.plant.num_contacts())
        # Create slack variable
        self.compl_slack = self.prog.NewContinuousVariables(1)

    def _add_costs(self):
        """
            Add costs to the program
            Currently only adds a cost on the feasibility / complementarity slack parameter
        """
        Q = np.eye(self.compl_slack.shape[0])
        b = np.zeros((self.compl_slack.shape[0]))
        self.slack_cost = self.prog.AddQuadraticErrorCost(Q, b, vars=self.compl_slack).evaluator().set_description("Feasibility Cost")

    def _add_constraints(self):
        """
            Add constraints to the program

            Added constraints include:
                Dynamics
                Normal Distance
                Sliding Velocity
                Friction Cone
                Friction Coefficient Nonnegativity
        """
        pass

    def _normal_distance(self):
        pass

    def _sliding_velocity(self):
        pass

    def _friction_cone(self):
        pass

    def add_distance_error_cost(self):
        """ Adds a quadratic cost on the distance error """
        Q = np.eye(self.dist_err.shape[0])
        b = np.zeros((self.dist_err.shape[0],))
        self.dist_cost = self.prog.AddQuadraticErrorCost(Q, b, vars=self.dist_err).evaluator().set_description("Distance Error Cost")

    def add_friction_error_cost(self):
        """ Adds a quadratic cost on the friction error"""
        Q = np.eye(self.fric_err.shape[0])
        b = np.zeros((self.fric_err.shape[0],))
        self.fric_cost = self.prog.AddQuadraticErrorCost(Q, b, vars=self.fric_err).evaluator().set_description("Friction Error Cost")

    def set_distance_cost_weight(self, weights=None):
        """ Set the weight the distance error cost"""
        if weights is not None:
            return

    def set_friction_cost_weight(self, weights=None):
        """ Set the weight in the friction error cost"""
        if weights is None:
            return

    def set_slack_cost_weight(self, weights=None):
        """ Set the weight in the feasibility cost"""
        if weights is None:
            return


    def estimate_terrain(self):
        pass

class KinematicsTerrainEstimator(TerrainEstimator):
    """
        KinematicsTerrainEstimator assumes only the state sequence (the kinematics) and control inputs are given.
        Reaction forces are not assumed given and are estimated as part of the solution procedure
    """
    def __init__(self, timesteppingplant=None):
        super(KinematicsTerrainEstimator, self).__init__(timesteppingplant)

    def _create_decision_variables(self):
        """ 
            Creates decision variables

            In KinematicsTerrainEstimator, create_decision_variables adds variables for reaction forces
        """
        # Create the usual variables
        super(KinematicsTerrainEstimator, self)._create_decision_variables()
        # Assume forces are unknown
        self.normal_force = self.prog.NewContinuousVariables(self.plant.num_contacts())
        self.sliding_vel = self.prog.NewContinuousVariables(self.plant.num_contacts())
        self.friction_force = self.prog.NewContinuousVariables(self.plant.num_friction())
        
    def _inverse_dynamics(self, x1, x2, u, h):
        """ 
        Calculates the generalized force required to achieve the next state
        
        Arguments:
            x1: State vector at time k
            x2: State vector at time k+1
            u:  Input vector
            h:  Timestep between times k and k+1

        Return values:
            tau: Vector of generalized forces required to achieve state x2 from state x1
        """
        # Split configuration and velocity
        _, v1 = np.split(x1,[self.plant.multibody.num_positions()])
        _, v2 = np.split(x2,[self.plant.multibody.num_positions()])
        dv = (v2 - v1)/h
        # Set the state as the future state
        context = self.plant.multibody.CreateDefaultContext()
        self.plant.multibody.SetPositionsAndVelocities(context, x2)
         # Calculate generalized forces
        B = self.plant.multibody.MakeActuationMatrix()
        forces = B.dot(u) + self.plant.multibody.CalcGravityGeneralizedForces(context)
        # Do inverse dynamics
        self.mbf.SetZero()
        tau = self.plant.multibody.CalcInverseDynamics(context, dv, self.mbf) - forces
        # Return the inverse dynamics force vector
        return tau

    def add_force_costs(self):
        """ Add a cost term of the reaction force magnitude """
        pass

    def set_force_cost_weights(self, weights=None):
        if weights is None:
            return

class TerrainEstimatorWithOrientation(TerrainEstimator):
    def __init__(self, timesteppingplant=None):
        super(TerrainEstimatorWithOrientation, self).__init__(timesteppingplant)

    def _normal_distance(self):
        pass

    def _sliding_velocity(self):
        pass

class ResidualTerrainEstimator():
    def __init__(self, timestepping_plant):
        self.plant = timestepping_plant
        self.mbf = MultibodyForces(timestepping_plant.multibody)
        self.context = self.plant.multibody.CreateDefaultContext()
        self._create_program()

    def _create_program(self):

        # Calculate the dimensions of the problem
        Jn, Jt, *_ = self.get_contact_parameters()
        numN = Jn.shape[0]
        numT = Jt.shape[0]
        numV = Jt.shape[1]
        # Create the program
        self.prog = MathematicalProgram()
        # Add decision variables
        self._create_decision_variables(numN, numT)
        # Add the costs and constraints
        self._add_costs(numN)
        self._add_constraints(numN, numT, numV)

    def _create_decision_variables(self, numN, numT):
        self.dist_err = self.prog.NewContinuousVariables(numN, name="terrain_residual")
        self.fric_err = self.prog.NewContinuousVariables(numN, name="friction_residual")
        self.fN = self.prog.NewContinuousVariables(numN, name="normal_forces")
        self.fT = self.prog.NewContinuousVariables(numT, name="friction_forces")
        self.gam = self.prog.NewContinuousVariables(numN, name="velocity_slack")

    def _add_costs(self, numN):
        """Add a quadratic cost on the terrain and friction residuals"""
        Q = 0.5 * np.eye(numN)
        b = np.zeros((numN,))
        self.prog.AddQuadraticErrorCost(Q,b,vars=self.dist_err).evaluator().set_description("DistanceErrCost")
        self.prog.AddQuadraticErrorCost(Q,b,vars=self.fric_err).evaluator().set_description("FrictionErrCost")

    def _add_constraints(self, numN, numT, numV):
        """add  feasibility constraints to the program"""        

        # Enforce feasibility of the contact forces
        self.dynamics = self.prog.AddLinearEqualityConstraint(
                            Aeq=np.zeros((numV, numN+numT)), 
                            beq=np.zeros((numV,)), 
                            vars=np.concatenate([self.fN, self.fT], axis=0))
        self.dynamics.evaluator().set_description("DynamicsConstraint")
        # Enforce complementarity in the residual model
        self.distance = NormalDistanceConstraint(phi = np.zeros((numN,)))
        self.velocity = SlidingVelocityConstraint(v = np.zeros((numT,)))
        self.friction = FrictionConeConstraint(mu = np.zeros((numN,)))
        # Add the constraints
        self.prog.AddConstraint(self.distance,
                            lb=np.concatenate([np.zeros((2*numN,)), -np.full((numN,), np.inf)], axis=0),
                            ub=np.concatenate([np.full((2*numN,), np.inf), np.zeros((numN,))], axis=0),
                            vars=np.concatenate((self.dist_err, self.fN), axis=0),
                            description="NormalDistanceConstraint")
        self.prog.AddConstraint(self.velocity,
                            lb=np.concatenate([np.zeros((2*numT,)), -np.full((numT,), np.inf)], axis=0),
                            ub=np.concatenate([np.full((2*numT,), np.inf), np.zeros((numT,))], axis=0),
                            vars=np.concatenate((self.fT, self.gam), axis=0),
                            description="SlidingVelocityConstraint")
        self.prog.AddConstraint(self.friction,
                            lb=np.concatenate([np.zeros((2*numN,)), -np.full((numN,), np.inf)], axis=0),
                            ub=np.concatenate([np.full((2*numN,), np.inf), np.zeros((numN,))], axis=0),
                            vars=np.concatenate((self.fric_err, self.fN, self.gam, self.fT),axis=0),
                            description="FrictionConeConstraint")
        # Ensure estimated friction is nonnegative
        self.bounding = self.prog.AddBoundingBoxConstraint(
                            np.zeros((numN,)), 
                            np.full((numN,), np.inf), 
                            self.fric_err)
        self.bounding.evaluator().set_description("FrictionErrorBoxConstraint")
    
    def _update_program(self, J, forces, mu, phi, vel):
        mu = np.asarray(mu)
        numN = mu.shape[0]
        # Update the dynamics constraint
        self.dynamics.evaluator().UpdateCoefficients(Aeq = J.transpose(), beq = forces)
        # Update the contact constraints
        self.distance.phi = phi
        self.velocity.vel = vel
        self.friction.mu = mu
        # Update the bounding box constraint on the friction coefficient
        self.bounding.evaluator().set_bounds(new_lb = -mu, new_ub = np.full((numN,), np.inf))

    def estimate_terrain(self, h, x1, x2, u):
        """Updates the terrain model in plant"""
        # Calculate the necessary forces to satisfy the dynamics
        f = self.do_inverse_dynamics(x1, x2, u, h)
        # Solve the optimization problem for the terrain residuals
        result = self.estimate_residuals(x2, f)
        # Update the terrain
        self.update_terrain(x2, result.GetSolution(self.dist_err))
        # Re-solve the optimization problem for the friction residuals (terrain shape might have changed)
        result = self.estimate_residuals(x2, f)
        # Update the friction coefficient
        self.update_friction(x2, result.GetSolution(self.fric_err))

    def update_friction(self, x2, fric_err):
        """Update the friction model in the plant"""
        # Collect contact points
        self.plant.multibody.SetPositionsAndVelocities(self.context, x2)
        pts, _  = self.plant.getTerrainPointsAndFrames(self.context)
        # Get the updated friction values
        mu = self.plant.GetFrictionCoefficients(self.context)
        new_mu = np.array(mu) + fric_err
        new_mu = np.expand_dims(new_mu, axis=1)
        pts = np.column_stack(pts)
        # Update the friction model
        self.plant.terrain.friction.add_data(pts[0:2,:], new_mu)

    def update_terrain(self, x2, dist_err):
        """Update the terrain model in the plant"""
        # Collect contact points
        self.plant.multibody.SetPositionsAndVelocities(self.context, x2)
        pts, frames = self.plant.getTerrainPointsAndFrames(self.context)
        # Update the sample points
        new_pts = np.zeros((pts[0].shape[0], len(pts)))
        for n in range(0, len(pts)):
            new_pts[:,n] = pts[n] - frames[n][0,:] * dist_err[n]
        # Add all the sample points to the terrain at the same time
        self.plant.terrain.height.add_data(new_pts[0:2,:], new_pts[2:,:])

    def do_inverse_dynamics(self, x1, x2, u, h):
        """ Calculates the generalized force required to achieve the next state"""
        # Split configuration and velocity
        _, v1 = np.split(x1,[self.plant.multibody.num_positions()])
        _, v2 = np.split(x2,[self.plant.multibody.num_positions()])
        dv = (v2 - v1)/h
        # Set the state as the future state
        context = self.plant.multibody.CreateDefaultContext()
        self.plant.multibody.SetPositionsAndVelocities(context, x2)
         # Calculate generalized forces
        B = self.plant.multibody.MakeActuationMatrix()
        forces = B.dot(u) + self.plant.multibody.CalcGravityGeneralizedForces(context)
        # Do inverse dynamics
        self.mbf.SetZero()
        tau = self.plant.multibody.CalcInverseDynamics(context, dv, self.mbf) - forces
        # Return the inverse dynamics force vector
        return tau

    def estimate_residuals(self, x2, forces):

        # Update the context
        self.plant.multibody.SetPositionsAndVelocities(self.context, x2)
        # Get the contact Jacobian
        Jn, Jt, phi, mu = self.get_contact_parameters()
        J = np.concatenate((Jn, Jt), axis=0)
        # Calculate the tangent velocity
        _, v = np.split(x2, [self.plant.multibody.num_positions()])
        dq = self.plant.multibody.MapVelocityToQDot(self.context, v)
        Vt = Jt.dot(dq)
        # Update the program
        self._update_program(J, forces, mu, phi, Vt)
        # Solve the program
        result = Solve(self.prog)
        # Get the solution from the variables
        return result

    def get_contact_parameters(self):
        Jn, Jt = self.plant.GetContactJacobians(self.context)
        phi = self.plant.GetNormalDistances(self.context)
        mu = self.plant.GetFrictionCoefficients(self.context)
        return (Jn, Jt, phi, mu)

    def unpack_solution(self, result):
        soln = {"dist_err": result.GetSolution(self.dist_err),
                "fric_err": result.GetSolution(self.fric_err),
                "fN": result.GetSolution(self.fN),
                "fT": result.GetSolution(self.fT),
                "gam": result.GetSolution(self.gam),
                "success": result.is_success(),
                "solver": result.get_solver_id().name(),
                "status": result.get_solver_details().info,
                "infeasible": result.GetInfeasibleConstraintNames(self.prog)
        }
        return soln

class NormalDistanceConstraint():
    def __init__(self, phi):
        self.phi = phi

    def __call__(self, z):
        err, fN = np.split(z,2)
        return np.concatenate((self.phi + err, fN, fN*(self.phi+err)), axis=0)

    @property
    def phi(self):
        return self.__phi 

    @phi.setter
    def phi(self, val):
        self.__phi = val

class SlidingVelocityConstraint():
    def __init__(self, v):
        self.vel = v
    
    def __call__(self, z):
        fT, gam = np.split(z, [self.vel.shape[0]])
        w = duplicator(gam.shape[0], fT.shape[0])
        r  = w.transpose().dot(gam) + self.vel
        return np.concatenate((r, fT, r * fT), axis=0)

    @property
    def vel(self):
        return self.__vel
    
    @vel.setter
    def vel(self, val):
        self.__vel = np.asarray(val)

class FrictionConeConstraint():
    def __init__(self, mu):
        self.mu = np.asarray(mu)
    
    def __call__(self, z):
        numN = self.mu.shape[0]
        err, fN, gam, fT = np.split(z, np.cumsum([numN, numN, numN]))
        w = duplicator(numN, fT.shape[0])
        r = np.diag(self.mu + err).dot(fN) - w.dot(fT)
        return np.concatenate((r, gam, r*gam), axis=0)

    @property
    def mu(self):
        return self.__mu 

    @mu.setter
    def mu(self, val):
        self.__mu = np.asarray(val)

def duplicator(numN, numT):

    w = np.zeros((numN, numT))
    nD = int(numT / numN)
    for n in range(0, numN):
        w[n, n*nD:(n+1)*nD] = 1
    return w

class ResidualTerrainEstimation_Debug(ResidualTerrainEstimator):
    """ Helper Class for debugging Residual Terrain Estimation"""
    def __init__(self, timesteppingplant):
        super(ResidualTerrainEstimation_Debug, self).__init__(timesteppingplant)
        self.pass1 = {"results": {},
                    "costs": {},
                    "constraints": {}
        }
        self.pass2 = {key: {} for key in self.pass1.keys()}

    def estimate_terrain(self, h, x1, x2, u):
        """Updates the terrain model in plant"""
        # Calculate the necessary forces to satisfy the dynamics
        f = self.do_inverse_dynamics(x1, x2, u, h)
        # Solve the optimization problem for the terrain residuals
        result = self.estimate_residuals(x2, f)
        # Store the results from estimation
        self.record_intermediate_values(result, self.pass1)
        # Update the terrain
        self.update_terrain(x2, result.GetSolution(self.dist_err))
        # Re-solve the optimization problem for the friction residuals (terrain shape might have changed)
        result = self.estimate_residuals(x2, f)
        # Store the results from the estimation
        self.record_intermediate_values(result, self.pass2)
        # Update the friction coefficient
        self.update_friction(x2, result.GetSolution(self.fric_err))

    def record_intermediate_values(self, result, storage_dict):
        # Calculate all cost values
        costs = self.evaluate_costs(result)
        # Calculate all constraint values
        cstrs = self.evaluate_constraints(result)
        # Record all costs, constraints, and solution values
        append_dict(costs, storage_dict["costs"])
        append_dict(cstrs, storage_dict["constraints"])
        append_dict(self.unpack_solution(result), storage_dict["results"])

    def evaluate_costs(self, soln):
        cvals = {}
        # Get all the costs in the program
        costs = self.prog.GetAllCosts()
        for cost in costs:
            # Use the cost description as a key
            name = cost.evaluator().get_description()
            # Evaluate the cost
            dvals = soln.GetSolution(cost.variables())
            cvals[name] = cost.evaluator().Eval(dvals)
        return cvals
            
    def evaluate_constraints(self, soln):
        cvals = {}
        # Get all the constraints in the program
        cstrs = self.prog.GetAllConstraints()
        for cstr in cstrs:
            # Use the constraint description as a key
            name = cstr.evaluator().get_description()
            # Evaluate the constraint
            dvals = soln.GetSolution(cstr.variables())
            cvals[name] = cstr.evaluator().Eval(dvals)
        return cvals

    def print_report(self):
        """ Print a debugging summary to the terminal"""
        # First pass
        print(f"In the first pass: ")
        self._print_report_for_pass(self.pass1)
        # Second pass
        print(f"In the second pass: ")
        self._print_report_for_pass(self.pass2)

    def _print_report_for_pass(self, pass_dict):
        """Print summaries for a given pass of the terrain estimation algorithm"""
        print(f"{sum(pass_dict['results']['success'])} of {len(pass_dict['results']['success'])} successful solves")
        print(f"The solvers used were: {set(pass_dict['results']['solver'])}")
        print(f"The exit codes were {set(pass_dict['results']['status'])}")
        infeas = [name for name in pass_dict['results']['infeasible'] if name]
        print(f"Infeasible constraints include {infeas}")
    
    def plot_constraints(self):
        """ Plot all recorded constraint values """
        self._plot_pass_constraints(self.pass1["constraints"], 1)
        self._plot_pass_constraints(self.pass2["constraints"], 2)
        plt.show()

    def _plot_pass_constraints(self, cstr, num):

        # Calculate number of plots from number of contacts
        numN = int(cstr["NormalDistanceConstraint"].shape[0]/3)
        numT = int(cstr["SlidingVelocityConstraint"].shape[0]/(3 * numN))
        for n in range(numN):
            _, axs = plt.subplots(2, 1)
            # Plot the normal distance
            plot_complementarity(axs[0], cstr["NormalDistanceConstraint"][n,:], cstr["NormalDistanceConstraint"][numN+n,:], 'Normal Distance','Normal Force')
            # Plot the friction cone
            plot_complementarity(axs[1], cstr["FrictionConeConstraint"][n,:], cstr["FrictionConeConstraint"][numN+n,:], "Friction Cone", "Sliding Slack")
            axs[0].set_title("Pass " +str(num) + " Contact Point " + str(n))
            _, axs2 = plt.subplots(numT, 1)
            # Plot the tangential forces
            vt = cstr["SlidingVelocityConstraint"][n*numT:(n+1)*numT,:]
            ft = cstr["SlidingVelocityConstraint"][(numN+n)*numT:(numN+n+1)*numT,:]
            for k in range(numT):
                plot_complementarity(axs2[k], ft[k,:], vt[k,:], 'Friction Force', 'Tangent Velocity')
            axs2[0].set_title("Pass " + str(num) + " Contact Point " + str(n))

    def plot_solutions(self):
        """ Plot all solution variables """
        pass

def append_dict(source, target):
    for key in source.keys():    
        if key in target and len(target[key]) > 0:
            if type(source[key]) is np.ndarray:
                if source[key].ndim == 1:
                    source[key] = np.expand_dims(source[key], axis=1)
                if target[key].ndim == 1:
                    target[key] = (np.expand_dims(target[key], axis=1))
                target[key] = np.concatenate((target[key], source[key]), axis=1)
            else:
                # Append the new value to the old one in a list
                target[key].append(source[key])
        elif type(source[key]) is np.ndarray:
            target[key] = source[key]
        else:
            # If the key doesn't exist or is empty, copy the data over
            target[key] = [source[key]]