"""
General constraint implementations for trajectory optimization

Luke Drnach
October 6, 2021
"""

import numpy as np
from trajopt.collocation import RadauCollocation
class RadauCollocationConstraint(RadauCollocation):
    #TODO: Double check the collocation constraint. Should only apply at (order) points  -DONE?
    def __init__(self, xdim, order):
        
        super(RadauCollocationConstraint, self).__init__(order, domain=[0, 1])
        self.xdim = xdim
        self.order = order
        self.continuity_weights = self.left_endpoint_weights()

    def addToProgram(self, prog, timestep, xvars, dxvars, x_final_last):
        prog = self._add_collocation(prog, timestep, xvars, dxvars)
        prog = self._add_continuity(prog, xvars, x_final_last)
        return prog

    def _add_collocation(self, prog, timestep, xvars, dxvars):
        # Add constraints on each element of the state vector separately to improve sparsity
        for n in range(self.xdim):
            dvars = np.concatenate([timestep, xvars[n,:], dxvars[n,1:]], axis=0)
            prog.AddConstraint(self._collocation_constraint, lb=np.zeros(self.order, ), ub=np.zeros(self.order, ), vars=dvars, description='CollocationConstraint')
        return prog

    def _add_continuity(self, prog, xvars, x_final_last):
        # Add linear constraints to each element of the state to improve sparsity
        aeq = np.append(self.continuity_weights, -1)
        for n in range(self.xdim):
            dvars = np.concatenate([xvars[n,:], x_final_last[n,:]], axis=0)
            prog.AddLinearEqualityConstraint(aeq, beq=np.zeros((1,)), vars=dvars, description='ContinuityConstraint')
        return prog

    def _collocation_constraint(self, dvars):
        # Apply the collocation constraint
        dt, x, dx = np.split(dvars, [1, 2+self.order])
        return dt * dx - self.differentiation_matrix[1:, :].dot(x)


class MultibodyConstraint():
    def __init__(self, plant_ad, plant_f):
        self.plant_ad = plant_ad
        self.plant_f = plant_f
        self.context_ad = self.plant_ad.multibody.CreateDefaultContext()
        self.context_f = self.plant_f.multibody.CreateDefaultContext()

    def _autodiff_or_float(self, z):
        if z.dtype == "float":
            return self.plant_f, self.context_f
        else:
            return self.plant_ad, self.context_ad

class MultibodyDynamicsConstraint(MultibodyConstraint):
    def __init__(self, plant_ad, plant_f):
        super(MultibodyDynamicsConstraint, self).__init__(plant_ad, plant_f)

    def upper_bound(self):
        return np.zeros((self.plant_ad.multibody.num_velocities(),))

    def lower_bound(self):
        return np.zeros((self.plant_ad.multibody.num_velocities(),))

    def eval(self, dvars):
        """
        eval: wrapper for _eval, supports a single input for use in MathematicalProgram
        
        """
        # Split the variables
        nx = self.plant_ad.multibody.num_positions()
        nv = self.plant_ad.multibody.num_velocities()
        nu = self.plant_ad.multibody.num_actuators()
        pos, vel, accel, control, force = np.split(dvars, np.cumsum([nx, nv, nv, nu]))
        # Evaluate multibody dynamics
        return self._eval(pos, vel, accel, control, force)

    def _eval(self, pos, vel, accel, control, force):
        plant, context = self._autodiff_or_float(pos)
        plant.multibody.SetPositionsAndVelocities(context, np.concatenate([pos, vel], axis=0))
        # Get the dynamics properties
        M = plant.multibody.CalcMassMatrixViaInverseDynamics(context)
        C = plant.multibody.CalcBiasTerm(context)
        G = plant.multibody.CalcGravityGeneralizedForces(context)
        B = plant.multibody.MakeActuationMatrix()
        Jn, Jt = plant.multibody.GetContactJacobians(context)
        # Integrated generalized force effects
        gen_forces = B.dot(control) - C + G
        # Integrate contact forces
        fN, fT = np.split(force, [plant.num_contacts()])
        Jl = plant.joint_limit_jacobian()
        if Jl is not None:
            fT, fL = np.split(fT, [plant.num_friction()])
            gen_forces += Jl.dot(fL)
        gen_forces += Jn.transpose().dot(fN) + Jt.transpose().dot(fT)
        # Do inverse dynamics
        return M.dot(accel) - gen_forces

