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
            new_pts[:,n] = pts[n] + frames[n][0,:] * soln["phi_res"][n]
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
        _, v = np.split(x2, [self.plant.multibody.num_positions()])
        dq = self.plant.multibody.MapVelocityToQDot(self.context, v)
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
        phi = self.plant.GetNormalDistances(context)
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
        prog.AddLinearEqualityConstraint(Aeq=J.transpose(), beq=f, vars=np.concatenate([dvars["fN"], dvars["fT"]], axis=0))
        # Enforce complementarity in the residual model
        prog.AddConstraint(NormalDistanceConstraint(phi=phi),
                            lb=np.concatenate([np.zeros((2*numN,)), -np.full((numN,), np.inf)], axis=0),
                            ub=np.concatenate([np.full((2*numN,), np.inf), np.zeros((numN,))], axis=0),
                            vars=np.concatenate((dvars["phi_res"], dvars["fN"]), axis=0))
        prog.AddConstraint(SlidingVelocityConstraint(v=Vt),
                            lb=np.concatenate([np.zeros((2*numT,)), -np.full((numT,), np.inf)], axis=0),
                            ub=np.concatenate([np.full((2*numT,), np.inf), np.zeros((numT,))], axis=0),
                            vars=np.concatenate((dvars["fT"], dvars["gam"]), axis=0))
        prog.AddConstraint(FrictionConeConstraint(mu=mu),
                            lb=np.concatenate([np.zeros((2*numN,)), -np.full((numN,), np.inf)], axis=0),
                            ub=np.concatenate([np.full((2*numN,), np.inf), np.zeros((numN,))], axis=0),
                            vars=np.concatenate((dvars["fric_res"], dvars["fN"], dvars["gam"], dvars["fT"]),axis=0))
        # Ensure estimated friction is nonnegative
        prog.AddBoundingBoxConstraint(-np.array(mu), np.full((numN,), np.inf), dvars["fric_res"])
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
        return soln
 
class NormalDistanceConstraint():
    def __init__(self, phi):
        self.phi = phi
    def __call__(self, z):
        err, fN = np.split(z,2)
        return np.concatenate((self.phi + err, fN, fN*(self.phi+err)), axis=0)

class SlidingVelocityConstraint():
    def __init__(self, v):
        self.v = v
    def __call__(self, z):
        fT, gam = np.split(z, [self.v.shape[0]])
        w = duplicator(gam.shape[0], fT.shape[0])
        r  = w.transpose().dot(gam) + self.v
        return np.concatenate((r, fT, r * fT), axis=0)

class FrictionConeConstraint():
    def __init__(self, mu):
        self.mu = np.asarray(mu)
    def __call__(self, z):
        numN = self.mu.shape[0]
        err, fN, gam, fT = np.split(z, np.cumsum([numN, numN, numN]))
        w = duplicator(numN, fT.shape[0])
        r = np.diag(self.mu + err).dot(fN) - w.dot(fT)
        return np.concatenate((r, gam, r*gam), axis=0)

def duplicator(numN, numT):
    w = np.zeros((numN, numT))
    nD = int(numT / numN)
    for n in range(0, numN):
        w[n, n*nD:(n+1)*nD] = 1
    return w