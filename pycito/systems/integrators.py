"""
integrators: integration schemes built on mathematical programming for timestepping.py

Luke Drnach
February 25, 2022
"""


import numpy as np

from pydrake.all import MathematicalProgram, SnoptSolver

from pycito.trajopt import constraints as cstr
from pycito.trajopt import complementarity as cp
import pycito.utilities as utils
class ContactDynamicsIntegrator():
    def __init__(self, plant, dynamics_class = cstr.BackwardEulerDynamicsConstraint, ncp=cp.NonlinearConstantSlackComplementarity):
        self.plant = plant
        self.dynamics = dynamics_class(self.plant)
        self.ncp = ncp
        self._state_constraint = None
        self._control_constraint = None
        self._timestep_constraint = None
        self.setup_program()
        # Set up the solver
        self.solver = SnoptSolver()
        self.solveroption = {'Major feasibility tolerance': 1e-6}
        # Store previous solves to warmstart the forces and slacks in the integrator
        self._last_force = None
        self._last_slack = None

    @classmethod
    def ImplicitEulerIntegrator(cls, plant, ncp = cp.NonlinearConstantSlackComplementarity):
        return cls(plant, dynamics_class = cstr.BackwardEulerDynamicsConstraint, ncp = ncp)

    @classmethod
    def SemiImplicitEulerIntegrator(cls, plant, ncp = cp.NonlinearConstantSlackComplementarity):
        return cls(plant, dynamics_class = cstr.SemiImplicitEulerDynamicsConstraint, ncp=ncp)

    @classmethod
    def ImplicitMidpointIntegrator(cls, plant, ncp=cp.NonlinearConstantSlackComplementarity):
        return cls(plant, dynamics_class = cstr.ImplicitMidpointDynamicsConstraint, ncp=ncp)

    def _add_dynamics_constraint(self):
        """Add the dynamics constraint to the program"""
        self.dynamics.addToProgram(self.prog, self.dt[:, 0], self.x[:, 0], self.x[:, 1], self.u[:, 0], self.all_forces[:, 0])

    def setup_program(self):
        """Create a mathematical program, adding variables and constraints"""
        self.prog = MathematicalProgram()
        self._create_variables()
        self._add_joint_limit_constraint()
        self._add_dynamics_constraint()
        self._add_contact_constraints()
        self._add_initial_conditions()
        
    def _create_variables(self):
        """Create the decision variables for the program"""
        self.x = self.prog.NewContinuousVariables(rows = self.plant.num_states, cols=2, name='states')
        self.u = self.prog.NewContinuousVariables(rows = self.plant.num_actuators, cols=1, name='controls')
        self.fn = self.prog.NewContinuousVariables(rows = self.plant.num_contacts(), cols=1, name='normal_forces')
        self.ft = self.prog.NewContinuousVariables(rows = self.plant.num_friction(), cols=1, name='friction_forces')
        self.vs = self.prog.NewContinuousVariables(rows = self.plant.num_contacts(), cols=1, name='relative_velocity')
        self.dt = self.prog.NewContinuousVariables(rows = 1, cols=1, name='timestep')

    def _add_initial_conditions(self):
        """Constraint the first state, the controls, and the timestep"""
        self._state_constraint = self.prog.AddBoundingBoxConstraint(np.zeros((self.plant.num_states,)), np.zeros((self.plant.num_states, )), self.x[:, 0])
        self._control_constraint = self.prog.AddBoundingBoxConstraint(np.zeros((self.plant.num_actuators, 1)), np.zeros((self.plant.num_actuators, 1)), self.u)
        self._timestep_constraint = self.prog.AddBoundingBoxConstraint(np.zeros((1,)), np.zeros((1,)), self.dt)

    def _add_contact_constraints(self):
        # Create the functions
        self.distance = cstr.NormalDistanceConstraint(self.plant)
        self.dissipation = cstr.MaximumDissipationConstraint(self.plant)
        self.friction = cstr.FrictionConeConstraint(self.plant)
        # Add the complementarity constraints
        self.distance_cstr = self.ncp(self.distance, 
                            xdim = self.plant.num_states, 
                            zdim = self.plant.num_contacts())
        self.distance_cstr.set_description('normal distance')
        self.distance_cstr.addToProgram(self.prog, self.x[:, 1], self.fn[:, 0])
        self.dissipation_cstr = self.ncp(self.dissipation, 
                            xdim=self.plant.num_states + self.plant.num_contacts(), 
                            zdim=self.plant.num_friction())
        self.dissipation_cstr.set_description('max dissipation')
        self.dissipation_cstr.addToProgram(self.prog, 
                            np.hstack([self.x[:, 1], 
                            self.vs[:, 0]]), 
                            self.ft[:, 0])
        self.friction_cstr = self.ncp(self.friction, 
                            xdim=self.plant.num_states + self.plant.num_contacts() + self.plant.num_friction(), 
                            zdim = self.plant.num_contacts())
        self.friction_cstr.set_description('friction cone')
        self.friction_cstr.addToProgram(self.prog,
                            np.hstack([self.x[:, 1], self.forces[:, 0]]),
                            self.vs[:, 0])
        # Add a normal dissipation constraint to constraint the size of the normal forces
        self.normal_dissipation = cstr.NormalDissipationConstraint(self.plant)
        self.normal_dissipation.addToProgram(self.prog, self.x[:, 1], self.fn[:, 0])

    def _add_joint_limit_constraint(self):
        """Add the joint limit constraint, if there are any"""
        if self.plant.has_joint_limits:
            self.jl = self.prog.NewContinuousVariables(rows = self.plant.num_joint_limits, cols=1, name='joint_limits')
            self.limits = cstr.JointLimitConstraint(self.plant)
            self.limit_cstr = self.ncp(self.limits, xdim=self.plant.num_positions, zdim=self.plant.num_joint_limits)
            self.limit_cstr.set_description('joint limits')
            self.limit_cstr.addToProgram(self.prog, 
                            self.x[:self.plant.num_positions, 1],
                            self.jl)
        else:
            self.jl = None
            self.limits = None
            self.limit_cstr = None

    def integrate(self, dt, x0, u):
        # Update the boundary constraints
        self._state_constraint.evaluator().UpdateUpperBound(x0)
        self._state_constraint.evaluator().UpdateLowerBound(x0)
        self._control_constraint.evaluator().UpdateLowerBound(u)
        self._control_constraint.evaluator().UpdateUpperBound(u)
        self._timestep_constraint.evaluator().UpdateLowerBound(dt)
        self._timestep_constraint.evaluator().UpdateUpperBound(dt)
        # Solve
        self.initialize(dt, x0, u)
        for key, value in self.solveroption.items():
            self.prog.SetSolverOption(self.solver.solver_id(), key, value)
        result = self.solver.Solve(self.prog)
        # Return the states, forces, and the status flag
        x = result.GetSolution(self.x[:, 1])
        f = result.GetSolution(self.forces)
        # Store the forces and slacks for the next iteration
        self._last_force = result.GetSolution(self.all_forces)
        self._last_slack = result.GetSolution(self.vs)
        if ~result.is_success():
            utils.printProgramReport(result, self.prog, terminal=True, verbose=True)
        return x, f, result.is_success()

    def initialize(self, dt, x0, u):
        """Initialize the decision variables in the program"""
        self.prog.SetInitialGuess(self.dt, dt)
        self.prog.SetInitialGuess(self.u, u)
        self.prog.SetInitialGuess(self.x[:, 0], x0)
        # assume constant velocity for the states
        context = self.plant.multibody.CreateDefaultContext()
        self.plant.multibody.SetPositionsAndVelocities(context, x0)
        q, v = np.split(x0, [self.plant.num_positions])
        q += dt * self.plant.multibody.MapVelocityToQDot(context, v)
        self.prog.SetInitialGuess(self.x[:, 1], np.hstack([q, v]))
        # Currently - do not initialize the forces (use previous solve if one exists)
        if self._last_force is not None:
            self.prog.SetInitialGuess(self.all_forces, self._last_force)
        else:
            # Initialize using the static case
            _, f = self.plant.static_controller(q)
            self.prog.SetInitialGuess(self.fn, f)
        if self._last_slack is not None:
            self.prog.SetInitialGuess(self.vs, self._last_slack)
        else:
            # Initialize using maximum velocity
            _, Jt = self.plant.GetContactJacobians(context)
            vmax = np.max(Jt.dot(v))
            self.prog.SetInitialGuess(self.vs, vmax * np.ones((self.plant.num_contacts(), )))

    @property
    def forces(self):
        return np.row_stack([self.fn, self.ft])

    @property
    def all_forces(self):
        if self.plant.has_joint_limits:
            return np.row_stack([self.fn, self.ft, self.jl])
        else:
            return self.forces

if __name__ == '__main__':
    print(f"Hello from integrators!")