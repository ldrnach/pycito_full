"""
terrainestimator: Terrain estimation algorithms

Luke Drnach
November 10, 2020
"""

import numpy as np
from pydrake.all import MathematicalProgram, MultibodyForces, Solve

#TODO: Write GaussianProcessTerrain with Update methods. Test on actual data

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
        self.prog.AddQuadraticCost(Q,b,vars=self.dist_err)
        self.prog.AddQuadraticCost(Q,b,vars=self.fric_err)

    def _add_constraints(self, numN, numT, numV):
        """add  feasibility constraints to the program"""        

        # Enforce feasibility of the contact forces
        self.dynamics = self.prog.AddLinearEqualityConstraint(
                            Aeq=np.zeros((numV, numN+numT)), 
                            beq=np.zeros((numV,)), 
                            vars=np.concatenate([self.fN, self.fT], axis=0))
        # Enforce complementarity in the residual model
        self.distance = NormalDistanceConstraint(phi = np.zeros((numN,)))
        self.velocity = SlidingVelocityConstraint(v = np.zeros((numT,)))
        self.friction = FrictionConeConstraint(mu = np.zeros((numN,)))
        # Add the constraints
        self.prog.AddConstraint(self.distance,
                            lb=np.concatenate([np.zeros((2*numN,)), -np.full((numN,), np.inf)], axis=0),
                            ub=np.concatenate([np.full((2*numN,), np.inf), np.zeros((numN,))], axis=0),
                            vars=np.concatenate((self.dist_err, self.fN), axis=0))
        self.prog.AddConstraint(self.velocity,
                            lb=np.concatenate([np.zeros((2*numT,)), -np.full((numT,), np.inf)], axis=0),
                            ub=np.concatenate([np.full((2*numT,), np.inf), np.zeros((numT,))], axis=0),
                            vars=np.concatenate((self.fT, self.gam), axis=0))
        self.prog.AddConstraint(self.friction,
                            lb=np.concatenate([np.zeros((2*numN,)), -np.full((numN,), np.inf)], axis=0),
                            ub=np.concatenate([np.full((2*numN,), np.inf), np.zeros((numN,))], axis=0),
                            vars=np.concatenate((self.fric_err, self.fN, self.gam, self.fT),axis=0))
        # Ensure estimated friction is nonnegative
        self.bounding = self.prog.AddBoundingBoxConstraint(
                            np.zeros((numN,)), 
                            np.full((numN,), np.inf), 
                            self.fric_err)

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
        soln = self.estimate_residuals(x2, f)
        # Update the terrain
        self.update_terrain(x2, soln)
        # Re-solve the optimization problem for the friction residuals (terrain shape might have changed)
        soln2 = self.estimate_residuals(x2, f)
        # Update the friction coefficient
        self.update_friction(x2, soln2)
        return (soln, soln2)

    def update_friction(self, x2, soln):
        """Update the friction model in the plant"""
        # Collect contact points
        self.plant.multibody.SetPositionsAndVelocities(self.context, x2)
        pts, _  = self.plant.getTerrainPointsAndFrames(self.context)
        # Get the updated friction values
        mu = self.plant.GetFrictionCoefficients(self.context)
        new_mu = np.array(mu) + soln["fric_err"]
        new_mu = np.expand_dims(new_mu, axis=1)
        pts = np.column_stack(pts)
        # Update the friction model
        self.plant.terrain.friction.add_data(pts[0:2,:], new_mu)

    def update_terrain(self, x2, soln):
        """Update the terrain model in the plant"""
        # Collect contact points
        self.plant.multibody.SetPositionsAndVelocities(self.context, x2)
        pts, frames = self.plant.getTerrainPointsAndFrames(self.context)
        # Update the sample points
        new_pts = np.zeros((pts[0].shape[0], len(pts)))
        for n in range(0, len(pts)):
            new_pts[:,n] = pts[n] - frames[n][0,:] * soln["dist_err"][n]
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
        return self.unpack_solution(result)

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
                "success": result.is_success()
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