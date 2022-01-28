"""
General constraint implementations for trajectory optimization

Luke Drnach
October 6, 2021
"""

import numpy as np
import abc

from pycito.trajopt.collocation import RadauCollocation

class RadauCollocationConstraint(RadauCollocation):
    #TODO: Double check the collocation constraint. Should only apply at (order) points  -DONE?
    def __init__(self, xdim, order):
        
        super(RadauCollocationConstraint, self).__init__(order, domain=[0, 1])
        self.xdim = xdim
        self.continuity_weights = self.left_endpoint_weights()

    def addToProgram(self, prog, timestep, xvars, dxvars, x_final_last):
        prog = self._add_collocation(prog, timestep, xvars, dxvars)
        prog = self._add_continuity(prog, xvars, x_final_last)
        return prog

    def _add_collocation(self, prog, timestep, xvars, dxvars):
        # Make sure the timestep is 2d
        timestep = np.atleast_2d(timestep)
        # Add constraints on each element of the state vector separately to improve sparsity
        for n in range(self.xdim):
            dvars = np.concatenate([timestep[0,:], xvars[n,:], dxvars[n,:-1]], axis=0)
            prog.AddConstraint(self._collocation_constraint, lb=np.zeros(self.order, ), ub=np.zeros(self.order, ), vars=dvars, description='CollocationConstraint')
        return prog

    def _add_continuity(self, prog, xvars, x_final_last):
        # Add linear constraints to each element of the state to improve sparsity
        aeq = np.expand_dims(np.append(self.continuity_weights, -1), axis=0)
        for n in range(self.xdim):
            dvars = np.expand_dims(np.concatenate([xvars[n,:], x_final_last[n,:]], axis=0), axis=1)
            prog.AddLinearEqualityConstraint(aeq, beq=np.zeros((1,1)), vars=dvars).evaluator().set_description('ContinuityConstraint')
        return prog

    def _collocation_constraint(self, dvars):
        # Apply the collocation constraint
        dt, x, dx = np.split(dvars, [1, 2+self.order])
        return dt * dx - self.differentiation_matrix[:-1, :].dot(x)

class MultibodyConstraint(abc.ABC):
    """
    Class template for implementing constraints related to multibody dynamics
    """
    def __init__(self, plant):
        self.plant = plant
        self._description = "multibody_constraint"

    @abc.abstractmethod
    def eval(self, dvals):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def lower_bound(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def upper_bound(self):
        raise NotImplementedError

    def _autodiff_or_float(self, z):
        """Check if we evaluate the multibody plant using autodiff or float type"""
        if z.dtype == "float":
            return self.plant, self.plant.multibody.CreateDefaultContext()
        else:
            plant_ad = self.plant.getAutoDiffXd()
            return plant_ad, plant_ad.multibody.CreateDefaultContext()

    def addToProgram(self, prog, *args):
        dvars = np.concatenate(args)
        prog.AddConstraint(self.eval, lb = self.lower_bound, ub=self.upper_bound, vars=dvars, description = self.description)
        return prog

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, text):
        self._description = str(text)

class MultibodyDynamicsConstraint(MultibodyConstraint):
    def __init__(self, plant):
        super(MultibodyDynamicsConstraint, self).__init__(plant)
        self._description = 'multibody_dynamics'

    @property
    def upper_bound(self):
        return np.zeros((self.plant.multibody.num_velocities(),))

    @property
    def lower_bound(self):
        return np.zeros((self.plant.multibody.num_velocities(),))

    def addToProgram(self, prog, pos, vel, accel, ctrl, force):
        dvars = np.concatenate([pos, vel, accel, ctrl, force])
        prog.AddConstraint(self.eval, lb = self.lower_bound, ub = self.upper_bound, vars=dvars, description='multibody_dynamics')
        return prog

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

    def _eval(self, plant, context, pos, vel, accel, control, force):
        plant, context = self._autodiff_or_float(pos)
        plant.multibody.SetPositionsAndVelocities(context, np.concatenate([pos, vel], axis=0))
        # Get the dynamics properties
        M = plant.multibody.CalcMassMatrixViaInverseDynamics(context)
        C = plant.multibody.CalcBiasTerm(context)
        G = plant.multibody.CalcGravityGeneralizedForces(context)
        B = plant.multibody.MakeActuationMatrix()
        Jn, Jt = plant.GetContactJacobians(context)
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

class BackwardEulerDynamicsConstraint(MultibodyConstraint):
    pass

class NormalDistanceConstraint(MultibodyConstraint):
    pass

class MaximumDissipationConstraint(MultibodyConstraint):
    pass

class FrictionConeConstraint(MultibodyConstraint):
    pass



if __name__ == '__main__':
    print('Hello from constraints.py!')