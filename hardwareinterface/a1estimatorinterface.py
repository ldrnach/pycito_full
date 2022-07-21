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
        self.estimator.useSnoptSolver()
        self.estimator.setSolverOptions(config['Solver'])
        # Store the slope estimate in case one iteration fails
        self.slope = np.zeros((3,))
        self.forces = np.zeros((4,))
        self.starttime = time.perf_counter()
        self.savecounter = 1
        print('Created A1 Contact Estimation Interface')

    def __del__(self):
        if self.logging_enabled:
            self.save_debug_logs()

    @staticmethod
    def _get_configuration():
        with open(CONFIGFILE, 'r') as file:
            config = json.load(file)
        return config

    @staticmethod
    def _calculate_ground_slope(model, yaw):
        """
        Calculate ground slope from the linear kernel model
        
        Based on the MIT code for calculating roll-pitch-yaw
        """
        null = np.zeros((3,))
        grad = model.surface.gradient(null)
        grad = np.squeeze(grad)

        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)]])
        coef = -1*grad[:2].T.dot(R.T)

        pitch = np.arctan2(coef[0], 1)
        roll = np.arctan2(coef[1], 1)
        offset = np.mean(model.surface._sample_points)
        yaw = grad[:1].dot(offset[:1])

        return np.array([roll, pitch, yaw])

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
        state = np.column_stack([position, rpy, joint_angles, velocity, angular_velocity, joint_velocities])
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
            self.logging_enabled = True
        else:
            self.logging_enabled=False

    def estimate(self, msg):
        """
        Estimate ground slope

        Arguments:
            msg: an lcm message containing the state, control torques, and timestamp of the A1 robot

        Return values:
            (float): the estimated ground slope        
        """
        # Keep track of time
        t = time.perf_counter() - self.starttime
        # Covert data in lcm message to numpy array
        x, u = self._lcm_to_arrays(msg)
        yaw = x[5]
        self.traj.append_sample(t, x, u)
        self.estimator.create_estimator()
        print(f'Estimating ground slope at time {t:.2f}')
        try:
            result = self.estimator.solve()
            if result.is_success():
                print('Estimation successful')
                self.estimator.update_trajectory(t, result)
                model = self.estimator.get_updated_contact_model(result)
                self.rpy = self._calculate_ground_slope(model, yaw)
                self.forces = self.estimator.traj._forces[-1]
            else:
                print('Estimation failed. Returning previous estimate')
        except:
            pass
        # Flush the estimator to prevent too much memory from being consumed
        if not self.logging_enabled:
            self.estimator.flush()
        elif t > 5 * self.savecounter:
            self.save_debug_logs()
            self.savecounter += 1
        return self.rpy, self.forces

    def save_debug_logs(self, directory=None):
        """Save the debugging logs from the estimator"""
        if directory is None:
            directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debuglogs')
        self.estimator.traj.save(os.path.join(directory, 'contact_trajectory.pkl'))
        self.estimator.logger.save(filename = os.path.join(directory, 'estimator_logs.pkl'))
        print(f"Logs saved to {directory}")

if __name__ == '__main__':
    print('Hello from A1 Estimator Interface!')