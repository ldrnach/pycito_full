"""
General constraint implementations for trajectory optimization

Luke Drnach
October 6, 2021
"""
import numpy as np
import abc

import pydrake.autodiffutils as ad

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

    def __call__(self, dvals):
        """Wrapper for _eval, supports single input needed to work with MathematicalProgram"""
        plant, context = self._autodiff_or_float(dvals)
        args = self.parse(dvals)
        return self.eval(plant, context, *args)

    @abc.abstractmethod
    def eval(self, dvals):
        raise NotImplementedError

    @abc.abstractmethod
    def parse(self, dvals):
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
        prog.AddConstraint(self, lb = self.lower_bound, ub=self.upper_bound, vars=dvars, description = self.description)
        return prog

    def linearize(self, *args):
        """
        returns a linearization of the constraint
        For the constraint function:
            g(x)
        The linearization returns the parameters (A, b) such that
            g(x + dx) ~= A*dx + b
        The parameters:
            b = g(x)
            A = dg/dx
        """
        dvals = np.concatenate(args)
        # Promote to AutoDiffType
        ad_vals = np.squeeze(ad.InitializeAutoDiff(dvals))
        fcn_ad = self(ad_vals)
        return ad.ExtractGradient(fcn_ad), np.reshape(ad.ExtractValue(fcn_ad), (-1,))

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

    def addToProgram(self, prog, pos, vel, accel, control, force):
        """Thin wrapper showing call syntax for MultibodyDynamicsConstraint.addToProgram"""
        return super(MultibodyDynamicsConstraint, self).addToProgram(prog, pos, vel, accel, control, force)

    @staticmethod
    def eval(plant, context, pos, vel, accel, control, force):
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

    def parse(self, dvals):
        """Split the decision variable list into state, control, and force"""
        # Get the sizes of each variable
        nx = self.plant.multibody.num_positions()
        nv = self.plant.multibody.num_velocities()
        nu = self.plant.multibody.num_actuators()
        # Split the variables (position, velocity, acceleration, control, external forces)
        return np.split(dvals, np.cumsum([nx, nv, nv, nu]))

class BackwardEulerDynamicsConstraint(MultibodyConstraint):
    def __init__(self, plant):
        super(BackwardEulerDynamicsConstraint, self).__init__(plant)
        self._description = "BE_dynamics"

    @property
    def upper_bound(self):
        return np.zeros((self.plant.multibody.num_positions() + self.plant.multibody.num_velocities(), ))

    @property
    def lower_bound(self):
        return np.zeros((self.plant.multibody.num_positions() + self.plant.multibody.num_velocities(), ))

    def addToProgram(self, prog, dt, state1, state2, control, force):
        """Thin wrapper showing call syntax for BackwardEulerDynamicsConstraint.addToProgram"""
        return super(BackwardEulerDynamicsConstraint, self).addToProgram(prog, dt, state1, state2, control, force)

    @staticmethod
    def eval(plant, context, dt, state1, state2, control, force):
        """
        Uses Backward Euler integration to evaluate the multibody plant dynamics

        Arguments:
            plant: The TimeSteppingMultibodyPlant model to operate on
            context: the associated MultibodyPlant Context
            dt: scalar, the integration timestep
            state1: the current state of the plant
            state2: the next state of the plant
            control: the control torques on the plant
            force: all external forces on the plant (contact forces and joint limits)

        Returns:
            an array of constraint defects, (pos_err, vel_err) containing the position integration error pos_err and the velocity integration error vel_err
        """
        # Get positions and velocities
        q1, v1 = np.split(state1, [plant.multibody.num_positions()])
        q2, v2 = np.split(state2, [plant.multibody.num_positions()])
        # Update the context - backward Euler integration
        plant.multibody.SetPositionsAndVelocities(context, state2)
        # Calculate the position integration error
        fq = q2 - q1 - dt*plant.multibody.MapVelocityToQDot(context, v2)
        # calculate generalized forces
        M = plant.multibody.CalcMassMatrixViaInverseDynamics(context)
        C = plant.multibody.CalcBiasTerm(context)
        G = plant.multibody.CalcGravityGeneralizedForces(context)
        B = plant.multibody.MakeActuationMatrix()
        # Integrated Generalized forces
        forces = (B.dot(control) - C + G)
        # External forces
        fN, fT = np.split(force, [plant.num_contacts()])
        # Joint limits
        if plant.has_joint_limits:
            fT, jl = np.split(fT, [plant.num_friction()])
            forces += plant.joint_limit_jacobian().dot(jl)
        # Contact reaction forces
        Jn, Jt = plant.GetContactJacobians(context)
        forces += Jn.transpose().dot(fN) + Jt.transpose().dot(fT)
        # Do inverse dynamics - velocity dynamics error
        fv = M.dot(v2 - v1) - dt*forces
        return np.concatenate((fq, fv), axis=0)
 
    def parse(self, dvals):
        """Split decision variables into states, controls, forces, etc"""
        # Get the state dimension
        nx = self.plant.multibody.num_positions() + self.plant.multibody.num_velocities()
        nu = self.plant.multibody.num_actuators()
        return np.split(dvals, np.cumsum([1, nx, nx, nu]))

class ExplicitEulerDynamicsConstraint(BackwardEulerDynamicsConstraint):
    def __init__(self, plant):
        super(ExplicitEulerDynamicsConstraint, self).__init__(plant)
        self._description = "explicit_dynamics"


    @staticmethod
    def eval(plant, context, dt, state1, state2, control, force):
        """
        Uses Backward Euler integration to evaluate the multibody plant dynamics

        Arguments:
            plant: The TimeSteppingMultibodyPlant model to operate on
            context: the associated MultibodyPlant Context
            dt: scalar, the integration timestep
            state1: the current state of the plant
            state2: the next state of the plant
            control: the control torques on the plant
            force: all external forces on the plant (contact forces and joint limits)

        Returns:
            an array of constraint defects, (pos_err, vel_err) containing the position integration error pos_err and the velocity integration error vel_err
        """
        # Get positions and velocities
        q1, v1 = np.split(state1, [plant.multibody.num_positions()])
        q2, v2 = np.split(state2, [plant.multibody.num_positions()])
        # Update the context - backward Euler integration
        plant.multibody.SetPositionsAndVelocities(context, state1)
        # Calculate the position integration error
        fq = q2 - q1 - dt*plant.multibody.MapVelocityToQDot(context, v1)
        # calculate generalized forces
        M = plant.multibody.CalcMassMatrixViaInverseDynamics(context)
        C = plant.multibody.CalcBiasTerm(context)
        G = plant.multibody.CalcGravityGeneralizedForces(context)
        B = plant.multibody.MakeActuationMatrix()
        # Integrated Generalized forces
        forces = (B.dot(control) - C + G)
        # External forces
        fN, fT = np.split(force, [plant.num_contacts()])
        # Joint limits
        if plant.has_joint_limits:
            fT, jl = np.split(fT, [plant.num_friction()])
            forces += plant.joint_limit_jacobian().dot(jl)
        # Contact reaction forces
        Jn, Jt = plant.GetContactJacobians(context)
        forces += Jn.transpose().dot(fN) + Jt.transpose().dot(fT)
        # Do inverse dynamics - velocity dynamics error
        fv = M.dot(v2 - v1) - dt*forces
        return np.concatenate((fq, fv), axis=0)
class SemiImplicitEulerDynamicsConstraint(BackwardEulerDynamicsConstraint):
    def __init__(self, plant):
        super(SemiImplicitEulerDynamicsConstraint, self).__init__(plant)
        self._description = "semi_implicit_dynamics"

    @staticmethod
    def eval(plant, context, dt, state1, state2, control, force):
        """
        Uses Backward Euler integration to evaluate the multibody plant dynamics

        Arguments:
            plant: The TimeSteppingMultibodyPlant model to operate on
            context: the associated MultibodyPlant Context
            dt: scalar, the integration timestep
            state1: the current state of the plant
            state2: the next state of the plant
            control: the control torques on the plant
            force: all external forces on the plant (contact forces and joint limits)

        Returns:
            an array of constraint defects, (pos_err, vel_err) containing the position integration error pos_err and the velocity integration error vel_err
        """
        # Get positions and velocities
        q1, v1 = np.split(state1, [plant.multibody.num_positions()])
        q2, v2 = np.split(state2, [plant.multibody.num_positions()])
        # Update the context - backward Euler integration
        plant.multibody.SetPositionsAndVelocities(context, state1)
        # Calculate the position integration error
        fq = q2 - q1 - dt*plant.multibody.MapVelocityToQDot(context, v2)
        # calculate generalized forces
        M = plant.multibody.CalcMassMatrixViaInverseDynamics(context)
        C = plant.multibody.CalcBiasTerm(context)
        G = plant.multibody.CalcGravityGeneralizedForces(context)
        B = plant.multibody.MakeActuationMatrix()
        # Integrated Generalized forces
        forces = (B.dot(control) - C + G)
        # External forces
        fN, fT = np.split(force, [plant.num_contacts()])
        # Joint limits
        if plant.has_joint_limits:
            fT, jl = np.split(fT, [plant.num_friction()])
            forces += plant.joint_limit_jacobian().dot(jl)
        # Contact reaction forces
        Jn, Jt = plant.GetContactJacobians(context)
        forces += Jn.transpose().dot(fN) + Jt.transpose().dot(fT)
        # Do inverse dynamics - velocity dynamics error
        fv = M.dot(v2 - v1) - dt*forces
        return np.concatenate((fq, fv), axis=0)
class ImplicitMidpointDynamicsConstraint(BackwardEulerDynamicsConstraint):
    def __init__(self, plant):
        super(ImplicitMidpointDynamicsConstraint, self).__init__(plant)
        self._description = "midpoint_dynamics"

    @staticmethod
    def eval(plant, context, dt, state1, state2, control, force):
        """
        Uses Backward Euler integration to evaluate the multibody plant dynamics

        Arguments:
            plant: The TimeSteppingMultibodyPlant model to operate on
            context: the associated MultibodyPlant Context
            dt: scalar, the integration timestep
            state1: the current state of the plant
            state2: the next state of the plant
            control: the control torques on the plant
            force: all external forces on the plant (contact forces and joint limits)

        Returns:
            an array of constraint defects, (pos_err, vel_err) containing the position integration error pos_err and the velocity integration error vel_err
        """
        # Get positions and velocities
        q1, v1 = np.split(state1, [plant.multibody.num_positions()])
        q2, v2 = np.split(state2, [plant.multibody.num_positions()])
        x_mid = (state1 + state2) / 2
        _, vmid = np.split(x_mid, [plant.multibody.num_positions()])
        # Update the context - backward Euler integration
        plant.multibody.SetPositionsAndVelocities(context, x_mid)
        # Calculate the position integration error
        fq = q2 - q1 - dt*plant.multibody.MapVelocityToQDot(context, vmid)
        # calculate generalized forces
        M = plant.multibody.CalcMassMatrixViaInverseDynamics(context)
        C = plant.multibody.CalcBiasTerm(context)
        G = plant.multibody.CalcGravityGeneralizedForces(context)
        B = plant.multibody.MakeActuationMatrix()
        # Integrated Generalized forces
        forces = (B.dot(control) - C + G)
        # External forces
        fN, fT = np.split(force, [plant.num_contacts()])
        # Joint limits
        if plant.has_joint_limits:
            fT, jl = np.split(fT, [plant.num_friction()])
            forces += plant.joint_limit_jacobian().dot(jl)
        # Contact reaction forces
        Jn, Jt = plant.GetContactJacobians(context)
        forces += Jn.transpose().dot(fN) + Jt.transpose().dot(fT)
        # Do inverse dynamics - velocity dynamics error
        fv = M.dot(v2 - v1) - dt*forces
        return np.concatenate((fq, fv), axis=0)

class NormalDistanceConstraint(MultibodyConstraint):
    """
    Implements a normal distance constraint. Ensures the normal contact distance is nonnegative.
    """
    
    def __init__(self, plant):
        super(NormalDistanceConstraint, self).__init__(plant)
        self._description = "normal_distance"

    @property
    def upper_bound(self):
        return np.full((self.plant.num_contacts(), ), np.inf)

    @property
    def lower_bound(self):
        return np.zeros((self.plant.num_contacts(), ))

    def addToProgram(self, prog, state):
        """Thin wrapper for call syntax for NormalDistanceConstraint.addToProgram"""
        return super(NormalDistanceConstraint, self).addToProgram(prog, state)

    @staticmethod
    def eval(plant, context, state):
        """Evaluate the normal contact distance"""
        # Calculate the normal distance
        plant.multibody.SetPositionsAndVelocities(context, state)    
        return plant.GetNormalDistances(context)

    def parse(self, dvals):
        """Returns the decision variable list"""
        return [dvals]

class MaximumDissipationConstraint(MultibodyConstraint):
    """
    Implements the maximum dissipation constraint (the sliding velocity portion)
    
    Maximum dissipation ensures that the contact point slides only when the maximum value of friction is reached.

    This constrain ensures only that the dissipation (the tangential sliding velocity) is nonnegative
    """
    def __init__(self, plant):
        super(MaximumDissipationConstraint, self).__init__(plant)
        self._description = 'max_dissipation'

    @property
    def upper_bound(self):
        return np.full((self.plant.num_friction(), ), np.inf)

    @property
    def lower_bound(self):
        return np.zeros((self.plant.num_friction(), ))

    def addToProgram(self, prog, state, slacks):
        """Thin wrapper showing call syntax for MaximumDissipationConstraint.addToProgram"""
        return super(MaximumDissipationConstraint, self).addToProgram(prog, state, slacks)

    @staticmethod
    def eval(plant, context, pos, vel, slacks):
        """Evaluate the relative sliding velocity"""
        plant.multibody.SetPositionsAndVelocities(context, np.concatenate([pos, vel], axis=0))
        # Get the contact Jacobian
        _, Jt = plant.GetContactJacobians(context)
        # Match sliding slacks to sliding velocities
        return plant.duplicator_matrix().T.dot(slacks) + Jt.dot(vel)

    def parse(self, dvals):
        """Parse and return the decision variables"""
        nq = self.plant.multibody.num_positions()
        nv = self.plant.multibody.num_velocities()
        return np.split(dvals, [nq, nq+nv])

class FrictionConeConstraint(MultibodyConstraint):
    """
    Implements a linearized friction cone constraint for multiple contact points.

    The friction cone constraint ensures that friction forces are nonnegative and do not exceed the maximum value (friction)*normal_force
    """
    
    def __init__(self, plant):
        super(FrictionConeConstraint, self).__init__(plant)
        self._description = "friction_cone"

    @property
    def upper_bound(self):
        """Upper bound of the friction cone constraint"""
        return np.full((self.plant.num_contacts(), ), np.inf)

    @property
    def lower_bound(self):
        """Lower bound of the friction cone constraint"""
        return np.zeros((self.plant.num_contacts(), ))

    def addToProgram(self, prog, state, normal_force, friction_force):
        """Thin wrapper showing call syntax for FrictionConeConstraint.addToProgram"""
        return super(FrictionConeConstraint, self).addToProgram(prog, state, normal_force, friction_force)

    @staticmethod
    def eval(plant, context, state, normal_force, friction_force):
        """Evaluate the linearized friction cone"""
        plant.multibody.SetPositionsAndVelocities(context, state)
        mu = plant.GetFrictionCoefficients(context)
        mu = np.diag(mu)
        # Evaluate linearized friction cone
        return mu.dot(normal_force) - plant.duplicator_matrix().dot(friction_force)

    def parse(self, dvals):
        """Split the decition variables into (state, normal_force, friction_force)"""
        nx = self.plant.multibody.num_positions() + self.plant.multibody.num_velocities()
        nf = self.plant.num_contacts()
        return np.split(dvals, np.cumsum([nx, nf]))

class JointLimitConstraint(MultibodyConstraint):
    def __init__(self, plant):
        super(JointLimitConstraint, self).__init__(plant)
        self._description = 'joint_limits'
        qmin = plant.multibody.GetPositionLowerLimits()
        qmax = plant.multibody.GetPositionUpperLimits()
        qmin_valid = np.isfinite(qmin)
        qmax_valid = np.isfinite(qmax)
        assert np.all(qmin_valid == qmax_valid), "One-sided joint limits not supported"
        self.num_joint_limits = np.sum(qmin_valid)

    @property 
    def upper_bound(self):
        return np.full((self.num_joint_limits, ), np.inf)

    @property
    def lower_bound(self):
        return np.zeros((self.num_joint_limits, ))

    def addToProgram(self, prog, qvars):
        return super(JointLimitConstraint, self).addToProgram(prog, qvars)

    @staticmethod
    def eval(plant, context, qvars):
        # Check which of the joints have limits
        qmin = plant.multibody.GetPositionLowerLimits()
        qmax = plant.multibody.GetPositionUpperLimits()
        qvalid = np.isfinite(qmin)
        return np.concatenate([qvars[qvalid] - qmin[qvalid], qmax[qvalid] - qvars[qvalid]], axis=0)

    def parse(self, dvals):
        return [dvals]

class LinearImplicitDynamics():
    """
    Just a linear constraint
    """
    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0], "A and b must have the same number of rows"
        self.A = A
        self.b = b

    def __eq__(self, obj):
        """Equality operator for LinearImplicitDynamics"""
        return type(self) is type(obj) and np.array_equal(self.A, obj.A) and np.array_equal(self.b, obj.b)

    def addToProgram(self, prog, *args):
        dvars = np.concatenate(args)
        assert self.A.shape[1] == dvars.shape[0], f"Expected {self.A.shape[1]} variables, but {dvars.shape[0]} were given"
        prog.AddLinearEqualityConstraint(Aeq = self.A, beq = -self.b, vars=dvars).evaluator().set_description('linear_dynamics')
        return prog

if __name__ == '__main__':
    print('Hello from constraints.py!')