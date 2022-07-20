import os
import numpy as np

from pycito.systems.A1.a1 import A1VirtualBase
from pycito.controller import mpc, contactestimator
from pycito.controller import mlcp as lcp

from pydrake.all import LeafSystem, BasicVector, Quaternion, AbstractValue, ContactResults
from pydrake.math import RollPitchYaw

REFERENCE = os.path.join('data','a1','reference','symmetric','3m','reftraj.pkl')

class A1StandingPDController(LeafSystem):
    """
    Basic controller interface for A1 quadruped robot


    The base controller implements a PD controller for A1 to stabilize it to a standing pose
    """

    def __init__(self, plant, dt):
        """
        Initialize the controller for the A1 Quadruped. The controller maps the current state to a control command, and logs the time, state and control for later analysis
        
        """
        LeafSystem.__init__(self)

        self.dt = dt
        # Store the plant and context from the DIAGRAM. Typically, this plant will use quaternions as a representation of orientation
        self.plant = plant
        self.context = self.plant.CreateDefaultContext()
        self.plant_ad = plant.ToAutoDiffXd()
        self.context_ad = self.plant_ad.CreateDefaultContext()

        # Declare the state as the input port
        self.DeclareVectorInputPort(
            "state",
            BasicVector(self.plant.num_positions() + self.plant.num_velocities())
        )
        # Declare the control as the output port
        self.DeclareVectorOutputPort(
            "torques",
            BasicVector(self.plant.num_actuators()),
            self.DoSetControlTorques
        )
        # Declare the output port for logging'
        self.DeclareVectorOutputPort(
            'logging',
            BasicVector(self.plant.num_positions() + self.plant.num_velocities() + self.plant.num_actuators()),
            self.SetLoggingOutputs
        )
        # Setup any internals in the controller
        self.control = np.zeros((self.plant.num_actuators(),))
        self._setup()

    def _setup(self):
        """
        Setup any internal data needed for the controller
        """
        # Create an internal A1 plant which uses Roll-Pitch-Yaw as the orientation representation
        self._internal_plant = A1VirtualBase()
        self._internal_plant.Finalize()
        # Set the position and velocity references
        q_guess = self._internal_plant.standing_pose()
        self.q_ref, _ = self._internal_plant.standing_pose_ik(base_pose = q_guess[:6], guess=q_guess)
        self.v_ref = np.zeros((self._internal_plant.multibody.num_velocities(),))

    def quaternion2rpy(self, q):
        """Convert the floating base quaternion into roll-pitch-yaw angles"""
        q = q / np.sqrt(np.sum(q**2))
        rpy = RollPitchYaw(Quaternion(q))
        return rpy.vector()

    def toVirtualPosition(self, q):
        """
        Convert the position vector for the simulator plant to the position vector for the internal plant (i.e. using the internal plant's virtual coordinate system)
        
        Converts quaternion orientation to roll-pitch-yaw to be more useful for control
        """
        quat, trans, joints = q[:4], q[4:7], q[7:]
        rpy = self.quaternion2rpy(quat)
        return np.concatenate([trans, rpy, joints], axis=0)
        
    def toVirtualVelocity(self, v):
        """
        Convert the velocity vector for the simulator plant to the velocity vector for the internal plant (e.g. using the internal plant's virtual coordinate system)

        Re-arranges the velocity components to the order: base-translation, base-rotation, joint-rotation        
        """
        rotation, translation, joints = v[:3], v[3:6], v[6:]
        return np.concatenate([translation, rotation, joints], axis=0)

    def MapQDotToVelocityVirtual(self, q, dq):
        """
        Map change in position coordinates to velocity coordinates, using the internal plant's virtual coordinate system
        """
        context = self._internal_plant.multibody.CreateDefaultContext()
        self._internal_plant.multibody.SetPositions(context, q)
        return self._internal_plant.multibody.MapQDotToVelocity(context, dq)

    def UpdateStoredContext(self, context):
        """
        Use the data in the given context to update self.context
        This is called at the beginning of each timestep
        """
        state = self.EvalVectorInput(context, 0).get_value()
        q = state[:self.plant.num_positions()]
        v = state[-self.plant.num_velocities():]

        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, v)

    def DoSetControlTorques(self, context, output):
        """
        Sends output torques to the simulator

        This function is called at every timestep in the simulation
        """
        self.UpdateStoredContext(context)
        # Get the state variables
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        # Calculate the control
        u = self.ControlLaw(context, q, v)
        # Send control output to drake
        output.SetFromVector(u)

    def ControlLaw(self, context, q, v):
        """
        Implements a PD control law to convert state to control torque
        
        This function is called by DoSetControlTorques and is the main control code for the robot
        """
        B = self.plant.MakeActuationMatrix()
        # Tuning parameters
        Kp = 60 * np.eye(self.plant.num_velocities())
        Kv = 0.3 * np.eye(self.plant.num_velocities())
        # Convert to internal position and velocity
        q = self.toVirtualPosition(q)
        v = self.toVirtualVelocity(q)
        # Compute errors
        q_err = self.MapQDotToVelocityVirtual(q, q - self.q_ref)
        v_err = v - self.v_ref
        # Compute desired generalized forces
        tau = -Kp.dot(q_err) - Kv.dot(v_err)
        self.control = B.transpose().dot(tau)
        return self.control

    def SetLoggingOutputs(self, context, output):
        """
        Set the outputs for logging.

        Outputs include:
            state
            control torque,
            forces        
        """
        q, v = self.plant.GetPositions(self.context), self.plant.GetVelocities(self.context)

        output.SetFromVector(np.concatenate([q, v, self.control], axis=0))


    def SetReference(self, q_ref, v_ref):
        """Set the position and velocity references"""
        assert q_ref.shape == self.q_ref.shape, 'Position reference is the wrong shape'
        assert v_ref.shape == self.v_ref.shape, 'Velocity reference is the wrong shape'
        self.q_ref = q_ref
        self.v_rev = v_ref


class A1ContactMPCController(A1StandingPDController):
    def __init__(self, 
                plant,
                dt,
                reference = REFERENCE,
                horizon = 17,
                lcptype = lcp.ConstantRelaxedPseudoLinearComplementarityConstraint):
        
        self.reference = reference
        self.horizon = horizon
        self.lcptype = lcptype
        self.control = []
        self.lasttimeindex = -1
        # Initialize        
        super().__init__(plant, dt)
        
    def _setup(self):
        """Setup the Linear Contact MPC"""
        # Create the reference trajectory
        a1 = A1VirtualBase()
        a1.terrain.friction = 1.0
        a1.Finalize()
        print('Loading reference trajectory...\t',end='', flush=True)
        reftraj = mpc.LinearizedContactTrajectory.loadLinearizedTrajectory(a1, self.reference)
        print('Done!', end='\n', flush=True)
        # Create the MPC Controller
        print('Creating MPC...', end='', flush=True)
        self.controller = mpc.LinearContactMPC(reftraj, self.horizon, self.lcptype)
        # Set the solver parameters for the MPC
        self._set_controller_options()
        print('Done!', end='\n', flush=True)

    def _set_controller_options(self):
        """
        Set the cost weights and solver options for the MPC controller

        #TODO: Refactor this into a configuration file
        """
        a1 = self.controller.lintraj.plant
        
        
        Kp = np.ones((a1.multibody.num_positions(),))
        Kv = np.ones((a1.multibody.num_velocities(), ))
        Kp[:6] = 1e2
        Kp[6:] = 1e2
        Ks = np.diag(np.concatenate([Kp, 1e-2*Kv], axis=0))   #1e2, 1e-2
        self.controller.statecost = Ks
        self.controller.controlcost = 1e-1*np.eye(self.controller.control_dim)    #1e-3
        self.controller.forcecost = 1e-4 * np.eye(self.controller.force_dim)      #1e-2
        self.controller.slackcost = 1e-4 * np.eye(self.controller.slack_dim)      #1e-4
        self.controller.limitcost = 1e-4 * np.eye(self.controller.jlimit_dim)     #1e-4
        #controller.complementarity_penalty = 1e-3
        self.controller.complementarity_schedule = [1e-2, 1e-4]    #originally 1e4
        self.controller.useSnoptSolver()
        self.controller.setSolverOptions({"Major feasibility tolerance": 1e-5,
                                    "Major optimality tolerance": 1e-5,
                                    'Scale option': 0,          #0
                                    'Major step limit':2.0,
                                    'Superbasics limit':1000,
                                    'Linesearch tolerance':0.9,
                                    'Iterations limit': 10000})
        self.controller.use_basis_file()
        self.controller.lintraj.useNearestTime()


    def ControlLaw(self, context, q, v):
        """Execute the MPC as the control law"""
        t = context.get_time()
        if self._check_time(t):
            # Convert to internal position and velocity
            q = self.toVirtualPosition(q)
            v = self.toVirtualVelocity(q)
            state = np.concatenate([q, v], axis=0)
            # Run MPC to get the control
            self.control = self.controller.get_control(t, state, self.control)
        return self.control
        
    def get_reference_trajectory(self):
        """Return the reference trajectory used by the controller"""
        return self.controller.lintraj

    def _check_time(self, t):
        return True
        # current_time = self.controller.lintraj.getTime(self.lasttimeindex + 1)
        # if current_time - t <= 1e-4:
        #     self.lasttimeindex += 1
        #     return True
        # else:
        #     return False

class BasicController(LeafSystem):
    """
    Basic controller for quadruped robot
    """

    def __init__(self, plant, dt):
        LeafSystem.__init__(self)

        self.dt = dt
        self.plant = plant
        self.context = self.plant.CreateDefaultContext()

        self.plant_ad = plant.ToAutoDiffXd()
        self.context_ad = self.plant_ad.CreateDefaultContext()

        # Declare input and output ports
        self.DeclareVectorInputPort(
            "state",
            BasicVector(self.plant.num_positions() + self.plant.num_velocities())
        )
        self.DeclareVectorOutputPort(
            "torques",
            BasicVector(self.plant.num_actuators()),
            self.DoSetControlTorques
        )

        self.q_ref = np.zeros((plant.num_positions(),))
        self.v_ref = np.zeros((plant.num_velocities(), ))
        self.control = np.zeros((plant.num_actuators(),))

        # Declare output port for logging
        self.DeclareVectorOutputPort(
            'output_metrics',
            BasicVector(plant.num_positions() + plant.num_velocities() + plant.num_actuators()),
            self.SetLoggingOutputs
        )


    def UpdateStoredContext(self, context):
        """
        Use the data in the given input context to update self.context.
        This is called at the beginning of each timestep
        """
        state = self.EvalVectorInput(context, 0).get_value()
        q = state[:self.plant.num_positions()]
        v = state[-self.plant.num_velocities():]

        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, v)

    def DoSetControlTorques(self, context, output):
        """
        This function is called at every timestep and sends output torques to the simulator
        """
        self.UpdateStoredContext(context)
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        u = self.ControlLaw(context, q, v)
        # Send control outputs to drake
        output.SetFromVector(u)

    def CalcDynamics(self):
        """
        Compute the dynamics quantities, M, Cv, tau_g, and S such that the robot's dynamics are given by:
            M(q)dv + C(q, v)v + tau_g = S*u + tau_ext
        Assumes self.context has been set properly
        """
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
        Cv = self.plant.CalcBiasTerm(self.context)
        tau_g = -self.plant.CalcGravityGeneralizedForces(self.context)
        S = self.plant.MakeActuationMatrix()

        return M, Cv, tau_g, S

    def ControlLaw(self, context, q, v):
        """
        This function is called by DoSetControlTorques and is the main control code for the robot
        """
        # Calculate dynamics
        M, C, N, B = self.CalcDynamics()
        # Tuning parameters
        Kp = 30*np.eye(self.plant.num_velocities())
        Kd = 1.5*np.eye(self.plant.num_velocities())

        # Compute desired generalized forces
        q_err = self.plant.MapQDotToVelocity(self.context, q - self.q_ref)
        dq_err = v - self.v_ref

        tau = -Kp.dot(q_err) - Kd.dot(dq_err)

        # Map generalized forces to control inputs 
        self.control = B.transpose().dot(tau)
        return self.control

    def SetLoggingOutputs(self, context, output):
        """
        Set outputs for logging
        """
        q, v = self.plant.GetPositions(self.context), self.plant.GetVelocities(self.context)
        output.SetFromVector(np.concatenate([q, v, self.control], axis=0))

    def SetReference(self, q_ref, v_ref):
        assert q_ref.shape == self.q_ref.shape, 'Position reference is the wrong shape'
        assert v_ref.shape == self.v_ref.shape, 'Velocity reference is the wrong shape'
        self.q_ref = q_ref
        self.v_rev = v_ref