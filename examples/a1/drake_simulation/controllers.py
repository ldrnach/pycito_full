import numpy as np

from pydrake.all import LeafSystem, BasicVector

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

        # Relevant frames

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