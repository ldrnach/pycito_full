"""
Description of A1 robot

Updated January 14, 2020
Includes classes for creating A1 MultibodyPlant and TimesteppingMultibodyPlant as well as an A1Visualizer class
"""
# Library imports
import numpy as np
from pydrake.all import PiecewisePolynomial
# Project Imports
from utilities import FindResource
from systems.timestepping import TimeSteppingMultibodyPlant
from systems.visualization import Visualizer
from systems.terrain import FlatTerrain

class A1(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/A1/A1_description/urdf/a1_foot_collision.urdf", terrain=FlatTerrain()):
        # Initialize the time-stepping multibody plant
        super(A1, self).__init__(file=FindResource(urdf_file), terrain=terrain)

    def print_frames(self, config=None):
        body_indices = self.multibody.GetBodyIndices(self.model_index)
        context = self.multibody.CreateDefaultContext()
        if config is not None:
            self.multibody.SetPositions(context, config)
        wframe = self.multibody.world_frame()
        for index in body_indices:
            body_name = self.multibody.get_body(index).name()
            body_frame = self.multibody.get_body(index).body_frame()
            print(f"A1 has a body called {body_name} with a frame called {body_frame.name()}")
            pW = self.multibody.CalcPointsPositions(context, body_frame, [0., 0., 0.], wframe)
            print(f"The origin of {body_frame.name()} in world coordinates is {pW.transpose()}")

    def configuration_sweep(self):
        """Create a visualization that sweeps through the configuration variables"""
        # Get the configuration vector
        context = self.multibody.CreateDefaultContext()
        pos = self.multibody.GetPositions(context)
        # Make a trajectory that sweeps each of the configuration variables
        # The expected order is (Orientation) (COM translation) (Leg Joints)
        t = np.linspace(0., 1., 101)
        angle = np.linspace(0., np.pi / 2, 101)
        trans = np.linspace(0., 1., 101)
        trajectory = PiecewisePolynomial.FirstOrderHold([0.,1.],np.column_stack((pos, pos)))
        pos = np.tile(pos, (101,1)).transpose()
        # Sweep the quaternions
        for n in range(1,4):
            pos_ = pos.copy()
            pos_[0,:] = np.cos(angle/2)
            pos_[n,:] = np.sin(angle/2)
            t_ = t + trajectory.end_time()
            trajectory.ConcatenateInTime(PiecewisePolynomial.FirstOrderHold(t_, pos_))
            
        # Sweep the COM translation
        for n in range(4,7):
            pos_ = pos.copy()
            pos_[n,:] = trans
            t_ = t + trajectory.end_time()
            trajectory.ConcatenateInTime(PiecewisePolynomial.FirstOrderHold(t_, pos_))

        # Sweep the joint positions
        for n in range(7,pos.shape[0]):
            pos_ = pos.copy()
            pos_[n,:] = angle
            t_ = t + trajectory.end_time()
            trajectory.ConcatenateInTime(PiecewisePolynomial.FirstOrderHold(t_, pos_))

        A1.visualize(trajectory)

    def get_joint_limits(self):
        return (self.multibody.GetPositionLowerLimits(), self.multibody.GetPositionUpperLimits())

    def get_velocity_limits(self):
        return (self.multibody.GetVelocityLowerLimits(), self.multibody.GetVelocityUpperLimits())

    def get_actuator_names(self):
        return ["FR_hip_motor","FR_thigh_motor","FR_calf_motor",
        "FL_hip_motor","FL_thigh_motor","FL_calf_motor",
        "RR_hip_motor","RR_thigh_motor","RR_calf_motor",
        "RL_hip_motor","RL_thigh_motor","RL_calf_motor"]

    def get_actuator_limits(self):
        act_names = self.get_actuator_names()
        act_limit = np.zeros((len(act_names),))
        for n in range(len(act_names)):
            act_limit[n] = self.multibody.GetJointActuatorByName(act_names[n]).effort_limit()
        return act_limit
       

    def standing_pose(self):
        # Get the default configuration vector
        context = self.multibody.CreateDefaultContext()
        pos = self.multibody.GetPositions(context)
        # Change the hip pitch to 45 degrees
        pos[11:15] = np.pi/4
        # Change the knee pitch to keep the foot under the hip
        pos[15:] = -np.pi/2
        # Adjust the base height to make the normal distances 0
        self.multibody.SetPositions(context, pos)
        dist = self.GetNormalDistances(context)
        pos[6] = pos[6] - np.amin(dist)
        return pos

    @staticmethod
    def visualize(trajectory=None):
        vis = Visualizer("systems/A1/A1_description/urdf/a1_no_collision.urdf")
        vis.visualize_trajectory(xtraj=trajectory)

if __name__ == "__main__":
    a1 = A1()
    a1.Finalize()
    print(f"A1 effort limits {a1.get_actuator_limits()}")
    qmin, qmax = a1.get_joint_limits()
    print(f"A1 has lower joint limits {qmin} and upper joint limits {qmax}")
    #a1.configuration_sweep()
    # a1.print_frames()
    # pos = a1.standing_pose()
    # pos2 = pos.copy()
    # pos2[4] = 1
    # traj = PiecewisePolynomial.FirstOrderHold(np.linspace(0.,1.,101),np.linspace(pos, pos2, 101, axis=1))
    # a1.visualize(traj)
    # print(f"The configuration is {pos}")
    # a1.print_frames(pos)
    
