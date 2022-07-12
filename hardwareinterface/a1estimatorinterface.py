"""
LCM Based interface for Contact Estimation on A1 Quadruped

Luke Drnach
July 11, 2022
"""
import os, json, sys
import numpy as np
import time

from pycito.systems.A1.a1 import A1VirtualBase
import pycito.controller.contactestimator as ce
import pycito.systems.contactmodel as cm
import pycito.systems.kernels as kernels

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lcmscripts'))

CONFIGFILE = os.path.join('hardwareinterface', 'estimatorconfig.json')

class A1ContactEstimationInterface():
    def __init__(self):
        """
        Setup the estimation interface for A1
        """
        config = self._get_configuration()
        self._make_a1_model(config)
        self._make_estimator(config['Estimator'])
        self.estimator.setSolverOptions(config['Solver'])
        # Store the slope estimate in case one iteration fails
        self.slope = 0.
        self.lasttime = time.perf_counter()
        print('Created A1 Contact Estimation Interface')

    @staticmethod
    def _get_configuration():
        with open(CONFIGFILE, 'r') as file:
            config = json.load(file)
        return config

    @staticmethod
    def _calculate_ground_slope(model):
        """
        Calculate ground slope from the linear kernel model
        """
        null = np.zeros((3,))
        grad = model.surface.gradient(null)
        return np.arctan2(grad[0,0], grad[0, 2])

    @staticmethod
    def _lcm_to_arrays(msg):
        """
        Get the timestamp, state data, and control signal from the lcm channel and return them as numpy arrays 
        """
        # Collect the state data
        joint_angles = np.column_stack(msg.q)
        joint_velocities = np.column_stack(msg.qd)
        position = np.column_stack(msg.p)
        velocity = np.column_stack(msg.vWorld)
        rpy = np.column_stack(msg.rpy)
        angular_velocity = np.column_stack(msg.omegaBody)
        state = np.row_stack([position, rpy, joint_angles, velocity, angular_velocity, joint_velocities])
        # Get the control signal
        control = np.column_stack(msg.tau_est)
        
        return np.squeeze(state), np.squeeze(control)
    
    def _generate_a1_initial_state(self, config):
        """
        Generate the initial state for A1
        """
        # Initial configuration
        q = np.zeros((self.a1.multibody.num_positions(),))
        q[6:9] = np.array(config['InitialLegPose'])
        q[9:12] = np.array(config['InitialLegPose'])
        q[12:15] = np.array(config['InitialLegPose'])
        q[15:] = np.array(config['InitialLegPose'])
        context = self.a1.multibody.CreateDefaultContext()
        self.a1.multibody.SetPositions(context, q)
        d = self.a1.GetNormalDistances(context)
        q[2] = np.abs(np.min(d))
        # Initial velocity
        v = np.zeros((self.a1.multibody.num_velocities(),))
        return np.concatenate([q,v], axis=0)

    def _make_a1_model(self, config):
        """Make the A1 model with a semiparametric contact model"""
        self.a1 = A1VirtualBase()
        frickernel = kernels.WhiteNoiseKernel(config['FrictionModel']['KernelRegularization'])
        surfkernel = kernels.RegularizedCenteredLinearKernel(
            weights = np.array(config['SurfaceModel']['KernelWeights']),
            noise = config['SurfaceModel']['KernelRegularization'])
        self.a1.terrain = cm.SemiparametricContactModel(
            surface = cm.SemiparametricModel(
                prior = cm.FlatModel(location = config['SurfaceModel']['PriorLocation'],
                                    direction = np.array(config['SurfaceModel']['PriorDirection'])),
                kernel = surfkernel
            ),
            friction = cm.SemiparametricModel(
                prior = cm.ConstantModel(config['FrictionModel']['Prior']),
                kernel = frickernel
            )
        )
        self.a1.Finalize()

    def _make_estimator(self, config):
        """
        Create the contact model estimator and set the cost weights appropriately
        """     
        initial_state = self._generate_a1_initial_state(config)   
        self.traj = ce.ContactEstimationTrajectory(self.a1, initial_state)
        self.estimator = ce.ContactModelEstimator(self.traj, horizon=config['Horizon'])
        # Set costs
        self.estimator.forcecost = config['ForceCost']
        self.estimator.relaxedcost = config['RelaxationCost']
        self.estimator.distancecost = config['DistanceCost']
        self.estimator.frictioncost = config['FrictionCost']
        self.estimator.velocity_scaling = config['VelocityScaling']
        self.estimator.force_scaling = config['ForceScaling']
        if config['EnableLogging']:
            self.estimator.enableLogging()

    def estimate(self, msg):
        """
        Estimate ground slope

        Arguments:
            msg: an lcm message containing the state, control torques, and timestamp of the A1 robot

        Return values:
            (float): the estimated ground slope        
        """
        # Keep track of time
        new_t = time.perf_counter()
        t, self.lasttime = np.array( new_t - self.lasttime), new_t
        # Covert data in lcm message to numpy array
        x, u = self._lcm_to_arrays(msg)
        self.traj.append_sample(t, x, u)
        self.estimator.create_estimator()
        print(f'Estimating ground slope at time {t:.2f}')
        result = self.estimator.solve()
        if result.is_success():
            print('Estimation successful')
            self.estimator.update_trajectory(t, result)
            model = self.estimator.get_updated_contact_model(result)
            self.slope = self._calculate_ground_slope(model)
        else:
            print('Estimation failed. Returning previous estimate')
        # Flush the estimator to prevent too much memory from being consumed
        self.estimator.flush()
        return self.slope

if __name__ == '__main__':
    print('Hello from A1 Estimator Interface!')