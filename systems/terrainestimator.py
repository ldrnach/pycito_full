"""
terrainestimator: Terrain estimation algorithms

Luke Drnach
November 10, 2020
"""

import numpy as np
from functools import partial
from pydrake.all import MathematicalProgram, MultibodyForces, Solve

#TODO: Write GaussianProcessTerrain with Update methods. Test on actual data

class ResidualTerrainEstimator():
    def __init__(self, timestepping_plant):
        self.plant = timestepping_plant
        self.mbf = MultibodyForces(timestepping_plant)
        self.context = self.plant.multibody.CreateDefaultContext()

    def estimate_terrain(self, h, x1, x2, u):
        """Updates the terrain model in plant"""
        # Calculate the necessary forces to satisfy the dynamics
        f = self.do_inverse_dynamics(x1, x2, u, h)
        # Solve the optimization problem for the terrain residuals
        soln = self.estimate_residuals(x2, f)
        # Update the terrain
        self.update_terrain(x2, soln)
        # Re-solve the optimization problem for the friction residuals (terrain shape might have changed)
        soln = self.estimate_residuals(x2, f)
        # Update the friction coefficient
        self.update_friction(x2, soln)

    def update_friction(self, x2, soln):
        """Update the friction model in the plant"""
        # Collect contact points
        self.plant.multibody.SetPositionsAndVelocities(self.context, x2)
        pts, _  = self.plant.getTerrainPointsAndFrames(self.context)
        # Get the updated friction values
        mu = self.plant.GetFrictionCoefficients(self.context)
        new_mu = np.array(mu) + soln["fric_res"]
        # Update the friction model
        for n in range(0, new_mu.shape[0]):
            self.plant.terrain.friction.addData(pts[0:1,n], new_mu[n])

    def update_terrain(self, x2, soln):
        """Update the terrain model in the plant"""
        # Collect contact points
        self.plant.multibody.SetPositionsAndVelocities(self.context, x2)
        pts, frames = self.plant.getTerrainPointsAndFrames(self.context)
        # Update the sample points
        new_pts = np.zeros(pts.shape)
        for n in range(0, pts.shape[1]):
            new_pts[:,n] = pts[:,n] + frames[0,:,n] * soln["phi_err"][n]
        # Add all the sample points to the terrain at the same time
        self.plant.terrain.addData(new_pts[0:1,:], new_pts[2,:])

    def do_inverse_dynamics(self, x1, x2, u, h):
        """ Calculates the generalized force required to achieve the next state"""
        # Split configuration and velocity
        _, v1 = np.split(x1,[self.plant.num_positions()])
        _, v2 = np.split(x2,[self.plant.num_positions()])
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

        # Setup the mathematical program
        prog, dvars = self.create_program(x2, forces)
        # Solve the program
        result = Solve(prog)
        # Get the solution from the variables
        return self.unpack_solution(result, dvars)

    def create_program(self, x2, forces):
        # First set the state of the system
        self.plant.multibody.SetPositionsAndVelocities(self.context, x2)
        # Calculate the dimensions of the problem
        Jn, Jt, phi, mu = self.get_contact_parameters(self.context)
        numN = Jn.shape[0]
        numT = Jt.shape[0]
        # Tangent velocity
        _, v = np.split(x2, [self.plant.num_positions()])
        dq = self.plant.MapVelocityToQDot(self.context, v)
        Vt = Jt.dot(dq)
        # Contact Jacobian
        J = np.concatenate([Jn, Jt], axis=0)
        # Create the program and all the variables
        prog = MathematicalProgram()
        dvars = self.setup_variables(prog, numN, numT)
        # Add the costs and constraints to the program
        prog = self.add_costs_and_constraints(prog, dvars, phi, J, forces, Vt, mu)
        return (prog, dvars)

    def get_contact_parameters(self, context):
        Jn, Jt = self.plant.GetContactJacobians(context)
        phi = self.plant.GetNormalDistance(context)
        mu = self.plant.GetFrictionCoefficients(context)
        return (Jn, Jt, phi, mu)

    @staticmethod
    def add_costs_and_constraints(prog, dvars, phi, J, f, Vt, mu):
        """add residual cost and feasibility constraints to the program"""
        numN = dvars["fN"].shape[0]
        numT = dvars["fT"].shape[0]
        # Add cost to choose the smallest residual that fits the data
        Q = 0.5 * np.eye(numN)
        b = np.zeros((numN,))
        prog.AddQuadraticCost(Q,b,vars=dvars["phi_res"])
        prog.AddQuadraticCost(Q,b,vars=dvars["fric_res"])
        # Enforce feasibility of the contact forces
        prog.AddLinearEqualityConstraint(J.transpose, f, vars=np.concatenate([dvars["fN"], dvars["fT"]], axis=0))
        # Enforce complementarity in the residual model
        prog.AddConstraint(partial(ResidualTerrainEstimator.normal_dist_constraint, phi=phi),
                            lb=np.concatenate([np.zeros((2*numN,)), -np.full((numN,), np.inf)], axis=0),
                            ub=np.concatentate([np.full((2*numN,), np.inf), np.zeros((numN,))], axis=0),
                            vars=np.concatenate((dvars["phi_res"], dvars["fN"]), axis=0))
        prog.AddConstraint(partial(ResidualTerrainEstimator.sliding_vel_constraint, v=Vt),
                            lb=np.concatenate([np.zeros((2*numT,)), -np.full((numT,), np.inf)], axis=0),
                            ub=np.concatentate([np.full((2*numT,), np.inf), np.zeros((numT,))], axis=0),
                            vars=np.concatenate((dvars["fT"], dvars["gam"]), axis=1))
        prog.AddConstraint(partial(ResidualTerrainEstimator.fric_cone_constraint, mu=mu),
                            lb=np.concatenate([np.zeros((2*numN,)), -np.full((numN,), np.inf)], axis=0),
                            ub=np.concatentate([np.full((2*numN,), np.inf), np.zeros((numN,))], axis=0),
                            vars=np.concatenate((dvars["fric_res"], dvars["fN"], dvars["gam"], dvars["fT"]),axis=1))
        # Ensure estimated friction is nonnegative
        prog.AddBoundingBoxConstraint(lb=-np.array(mu), ub=np.full((numN,), np.inf), vars=dvars["fric_res"])
        # Return the updated program
        return prog

    @staticmethod
    def setup_variables(prog, numN, numT):
        dvars = {}
        dvars["phi_res"] = prog.NewContinuousVariables(numN, name="terrain_residual")
        dvars["fric_res"] = prog.NewContinuousVariables(numN, name="friction_residual")
        dvars["fN"] = prog.NewContinuousVariables(numN, name="normal_forces")
        dvars["fT"] = prog.NewContinuousVariables(numT, name="friction_forces")
        dvars["gam"] = prog.NewContinuousVariables(numN, name="velocity_slack")
        return dvars

    @staticmethod
    def unpack_solution(result, dvars):
        soln = {}
        for key in dvars.keys():
            soln[key] = result.GetSolution(dvars[key])
        soln["Success"] = result.is_success()
 
    @staticmethod
    def normal_dist_constraint(phi, z):
        err, fN = np.split(z,2)
        return (phi + err, fN, fN*(phi+err))

    @staticmethod
    def sliding_vel_constraint(v, z):
        fT, gam = np.split(z, [v.shape[0]])
        w = ResidualTerrainEstimator.duplicator(gam.shape[0], fT.shape[0])
        r  = w.transpose().dot(gam) + v
        return (r, fT, r * fT)

    @staticmethod
    def fric_cone_constraint(mu, z):
        numN = mu.shape[0]
        err, fN, gam, fT = np.split(z, [numN, numN, numN])
        w = ResidualTerrainEstimator.duplicator(numN, fT.shape[0])
        r = np.diag(mu + err).dot(fN) - w.dot(fT)
        return (r, gam, r*gam)

    @staticmethod
    def duplicator(numN, numT):
        w = np.zeros((numN, numT))
        nD = int(numT / numN)
        for n in range(0, numN):
            w[n, n*nD:(n+1)*nD] = 1
        return w