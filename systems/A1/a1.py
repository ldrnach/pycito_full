"""
Description of A1 robot

Updated January 14, 2020
Includes classes for creating A1 MultibodyPlant and TimesteppingMultibodyPlant as well as an A1Visualizer class
"""
# Library imports
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import PiecewisePolynomial, InverseKinematics, Solve, Body, PrismaticJoint, BallRpyJoint, SpatialInertia, UnitInertia
# Project Imports
from utilities import FindResource, GetKnotsFromTrajectory, quat2rpy
from systems.timestepping import TimeSteppingMultibodyPlant
from systems.visualization import Visualizer
from systems.terrain import FlatTerrain
from trajopt.quatutils import rpy2quat

class A1(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/A1/A1_description/urdf/a1_foot_collision.urdf", terrain=FlatTerrain()):
        # Initialize the time-stepping multibody plant
        super(A1, self).__init__(file=FindResource(urdf_file), terrain=terrain)

    def useFloatingRPYJoint(self):
        if self.multibody.is_finalized():
            raise RuntimeError("useFloatingRPYJoint must be called before finalize")
        zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
        # Create virtual, zero-mass
        xlink = self.multibody.AddRigidBody('xlink', self.model_index, zeroinertia)
        ylink = self.multibody.AddRigidBody('ylink', self.model_index, zeroinertia)
        zlink = self.multibody.AddRigidBody('zlink', self.model_index, zeroinertia)
        # Create the translational and rotational joints
        xtrans = PrismaticJoint("xtranslation", self.multibody.world_frame(), xlink.body_frame(), [1., 0., 0.])
        ytrans = PrismaticJoint("ytranslation", xlink.body_frame(), ylink.body_frame(), [0., 1., 0.])
        ztrans = PrismaticJoint("ztranslation", ylink.body_frame(), zlink.body_frame(), [0., 0., 1.])
        rpyrotation = BallRpyJoint("baseorientation", zlink.body_frame(), self.multibody.GetBodyByName('base').body_frame())
        # Add the joints to the multibody plant
        self.multibody.AddJoint(xtrans)
        self.multibody.AddJoint(ytrans)
        self.multibody.AddJoint(ztrans)
        self.multibody.AddJoint(rpyrotation)

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

    def standing_pose_ik(self, base_pose, guess=None):
        # Create an IK Problem
        IK = InverseKinematics(self.multibody, with_joint_limits=True)
        # Get the context and set the default pose
        context = self.multibody.CreateDefaultContext()
        q_0 = np.zeros((self.multibody.num_positions(),))
        q_0[0:7] = base_pose
        self.multibody.SetPositions(context, q_0)
        # Constrain the foot positions
        world = self.multibody.world_frame()
        for pose, frame, radius in zip(self.collision_poses, self.collision_frames, self.collision_radius):
            point = pose.translation().copy()
            point[-1] -= radius
            point_w = self.multibody.CalcPointsPositions(context, frame, point, world)
            point_w[-1] = 0.
            IK.AddPositionConstraint(frame, point, world, point_w, point_w)
        # Set the base position as a constraint
        q_vars = IK.q()
        prog = IK.prog()
        prog.AddLinearEqualityConstraint(Aeq = np.eye(7), beq = np.expand_dims(base_pose, axis=1), vars=q_vars[0:7])
        # Solve the problem
        prog = IK.prog()
        if guess is not None:
            # Set the initial guess
            prog.SetInitialGuess(q_vars, guess)
        result = Solve(prog)
        # Return the configuration vector
        return result.GetSolution(IK.q()), result.is_success()

    def plot_trajectories(self, xtraj=None, utraj=None, ftraj=None, jltraj=None):
        """ Plot all the trajectories for A1 """
        show_all = False
        if xtraj:
            self.plot_state_trajectory(xtraj, show=False)
            show_all = True
        if utraj:
            self.plot_control_trajectory(utraj, show=False)
            show_all = True
        if ftraj:
            self.plot_force_trajectory(ftraj, show=False)
            show_all = True
        if jltraj:
            self.plot_limit_trajectory(jltraj, show=False)
            show_all = True
        if show_all:
            plt.show()

    def plot_state_trajectory(self, xtraj, show=True):
        """ Plot the state trajectory for A1"""
        # Get the configuration and velocity trajectories as arrays
        t, x = GetKnotsFromTrajectory(xtraj)
        nq = self.multibody.num_positions()
        q, v = np.split(x, [nq])
        # Get orientation from quaternion
        q[1:4] = quat2rpy(q[0:4,:])
        # Plot COM orientation and position
        _, paxs = plt.subplots(2,1)
        labels=[["Roll", "Pitch", "Yaw"],["X", "Y", "Z"]]
        ylabels = ["Orientation","Position"]
        for n in range(2):
            for k in range(3):
                paxs[n].plot(t, q[1 + 3*n + k,:], linewidth=1.5, label=labels[n][k])
            paxs[n].set_ylabel(ylabels[n])
            paxs[n].legend()
        paxs[-1].set_xlabel('Time (s)')
        paxs[0].set_title("COM Configuration")
        #Plot COM orientation rate and translational velocity
        _, axs = plt.subplots(2,1)
        for n in range(2):
            for k in range(3):
                axs[n].plot(t, v[3*n + k,:], linewidth=1.5, label=labels[n][k])
            axs[n].set_ylabel(ylabels[n] + " Rate")
            axs[n].legend()
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("COM Velocities")
        # Plot joint positions and velocities 
        _, jaxs = plt.subplots(3,1)
        _, jvaxs = plt.subplots(3,1)
        # Loop over each joint angle
        legs = ["FR", "FL", "BR", "BL"]
        angles = ["Hip Roll", "Hip Pitch","Knee Pitch"]
        for n in range(3):
            # Loop over each leg
            for k in range(4):
                jaxs[n].plot(t, q[7+4*n+k,:], linewidth=1.5, label=legs[k])
                jvaxs[n].plot(t, v[6+4*n+k, :], linewidth=1.5, label=legs[k])
            jaxs[n].set_ylabel(angles[n])
            jvaxs[n].set_ylabel(angles[n] + " Rate")
        jaxs[-1].set_xlabel('Time (s)')
        jvaxs[-1].set_xlabel('Time (s)')
        jaxs[0].set_title("Joint Angles")
        jvaxs[0].set_title("Joint Rates")
        jaxs[0].legend()
        jvaxs[0].legend()            
        if show:
            plt.show()

    def plot_control_trajectory(self, utraj, show=True):
        """Plot joint actuation torque trajectories"""
        # Get the knot points from the trajectory
        t, u = GetKnotsFromTrajectory(utraj)
        # Plot the actuation torques, organized by joint angle
        _, axs = plt.subplots(3,1)
        leg = ['FR','FL','BR','BL']
        angles = ['Hip Roll','Hip Pitch','Knee Pitch']
        for n in range(3):
            for k in range(4):
                axs[n].plot(t, u[n+3*k,:], linewidth=1.5, label=leg[k])
            axs[n].set_ylabel(angles[n])
        # Set x label and title
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("Joint Actuation Torques")   
        axs[0].legend()
        # Show the plot 
        if show:
            plt.show()

    def plot_force_trajectory(self, ftraj, show=True):
        """ Plot reaction force trajectories"""
        t, f = GetKnotsFromTrajectory(ftraj)
        f = self.resolve_forces(f)  
        _, axs = plt.subplots(3,1)
        legs = ['FR', 'FL', 'BR', 'BL']
        labels = ['Normal', 'Friction-1', 'Friction-2']
        for k in range(3):
            for n in range(4):
                axs[k].plot(t, f[n + 4*k,:], linewidth=1.5, label=legs[n])
            axs[k].set_ylabel(labels[k])
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title('Reaction Forces')
        axs[0].legend()
        if show:
            plt.show()

    def plot_limit_trajectory(self, jltraj, show=True):
        """
        Plot the joint limit torque trajectories
        """
        t, jl = GetKnotsFromTrajectory(jltraj)
        jl = self.resolve_limit_forces(jl)
        leg = ['FR','FL','BR','BL']
        angle = ['Hip Roll','Hip Pitch','Knee Pitch']
        _, axs = plt.subplots(3,1)
        for n in range(3):
            for k in range(4):
                axs[n].plot(t, jl[4*n + k,:], linewidth=1.5, label=leg[k])
            axs[n].set_ylabel(angle[n])
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title('Joint Limit Torques')
        axs[0].legend()
        if show:
            plt.show()

    @staticmethod
    def visualize(trajectory=None):
        vis = Visualizer("systems/A1/A1_description/urdf/a1_no_collision.urdf")
        vis.visualize_trajectory(xtraj=trajectory)

class A1VirtualBase(A1):
    def __init__(self, urdf_file="systems/A1/A1_description/urdf/a1_foot_collision.urdf", terrain=FlatTerrain()):
        super(A1VirtualBase, self).__init__(urdf_file, terrain)
        # Add in virtual joints to represent the floating base
        zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
        # Create virtual, zero-mass
        xlink = self.multibody.AddRigidBody('xlink', self.model_index[0], zeroinertia)
        ylink = self.multibody.AddRigidBody('ylink', self.model_index[0], zeroinertia)
        zlink = self.multibody.AddRigidBody('zlink', self.model_index[0], zeroinertia)
        # Create the translational and rotational joints
        xtrans = PrismaticJoint("xtranslation", self.multibody.world_frame(), xlink.body_frame(), [1., 0., 0.])
        ytrans = PrismaticJoint("ytranslation", xlink.body_frame(), ylink.body_frame(), [0., 1., 0.])
        ztrans = PrismaticJoint("ztranslation", ylink.body_frame(), zlink.body_frame(), [0., 0., 1.])
        rpyrotation = BallRpyJoint("baseorientation", zlink.body_frame(), self.multibody.GetBodyByName('base').body_frame())
        # Add the joints to the multibody plant
        self.multibody.AddJoint(xtrans)
        self.multibody.AddJoint(ytrans)
        self.multibody.AddJoint(ztrans)
        self.multibody.AddJoint(rpyrotation)
    
    def standing_pose(self):
        # Get the default configuration vector
        context = self.multibody.CreateDefaultContext()
        pos = self.multibody.GetPositions(context)
        # Change the hip pitch to 45 degrees
        pos[10:14] = np.pi/4
        # Change the knee pitch to keep the foot under the hip
        pos[14:] = -np.pi/2
        # Adjust the base height to make the normal distances 0
        self.multibody.SetPositions(context, pos)
        dist = self.GetNormalDistances(context)
        pos[2] = pos[2] - np.amin(dist)
        return pos

    def standing_pose_ik(self, base_pose, guess=None):
        """

        base_pose should be specified as [xyz, rpy], a (6,) array
        """
        # Create an IK Problem
        IK = InverseKinematics(self.multibody, with_joint_limits=True)
        # Get the context and set the default pose
        context = self.multibody.CreateDefaultContext()
        q_0 = np.zeros((self.multibody.num_positions(),))
        q_0[0:6] = base_pose
        self.multibody.SetPositions(context, q_0)
        # Constrain the foot positions
        world = self.multibody.world_frame()
        for pose, frame, radius in zip(self.collision_poses, self.collision_frames, self.collision_radius):
            point = pose.translation().copy()
            point[-1] -= radius
            point_w = self.multibody.CalcPointsPositions(context, frame, point, world)
            point_w[-1] = 0.
            IK.AddPositionConstraint(frame, point, world, point_w, point_w)
        # Set the base position as a constraint
        q_vars = IK.q()
        prog = IK.prog()
        prog.AddLinearEqualityConstraint(Aeq = np.eye(6), beq = np.expand_dims(base_pose, axis=1), vars=q_vars[0:6])
        # Solve the problem
        prog = IK.prog()
        if guess is not None:
            # Set the initial guess
            prog.SetInitialGuess(q_vars, guess)
        result = Solve(prog)
        # Return the configuration vector
        return result.GetSolution(IK.q()), result.is_success()

    def plot_state_trajectory(self, xtraj, show=True):
        """ Plot the state trajectory for A1"""
        # Get the configuration and velocity trajectories as arrays
        t, x = GetKnotsFromTrajectory(xtraj)
        nq = self.multibody.num_positions()
        q, v = np.split(x, [nq])
        # Get orientation from quaternion
        # Plot COM orientation and position
        _, paxs = plt.subplots(2,1)
        labels=[["X", "Y", "Z"],["Roll", "Pitch", "Yaw"]]
        ylabels = ["Position","Orientation"]
        for n in range(2):
            for k in range(3):
                paxs[n].plot(t, q[3*n + k,:], linewidth=1.5, label=labels[n][k])
            paxs[n].set_ylabel(ylabels[n])
            paxs[n].legend()
        paxs[-1].set_xlabel('Time (s)')
        paxs[0].set_title("COM Configuration")
        #Plot COM orientation rate and translational velocity
        _, axs = plt.subplots(2,1)
        for n in range(2):
            for k in range(3):
                axs[n].plot(t, v[3*n + k,:], linewidth=1.5, label=labels[n][k])
            axs[n].set_ylabel(ylabels[n] + " Rate")
            axs[n].legend()
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("COM Velocities")
        # Plot joint positions and velocities 
        _, jaxs = plt.subplots(3,1)
        _, jvaxs = plt.subplots(3,1)
        # Loop over each joint angle
        legs = ["FR", "FL", "BR", "BL"]
        angles = ["Hip Roll", "Hip Pitch","Knee Pitch"]
        for n in range(3):
            # Loop over each leg
            for k in range(4):
                jaxs[n].plot(t, q[6+4*n+k,:], linewidth=1.5, label=legs[k])
                jvaxs[n].plot(t, v[6+4*n+k, :], linewidth=1.5, label=legs[k])
            jaxs[n].set_ylabel(angles[n])
            jvaxs[n].set_ylabel(angles[n] + " Rate")
        jaxs[-1].set_xlabel('Time (s)')
        jvaxs[-1].set_xlabel('Time (s)')
        jaxs[0].set_title("Joint Angles")
        jvaxs[0].set_title("Joint Rates")
        jaxs[0].legend()
        jvaxs[0].legend()            
        if show:
            plt.show()


if __name__ == "__main__":
    a1 = A1()
    a1.Finalize()
    print(f"A1 effort limits {a1.get_actuator_limits()}")
    qmin, qmax = a1.get_joint_limits()
    print(f"A1 has lower joint limits {qmin} and upper joint limits {qmax}")
    print(f"A1 has actuation matrix:")
    print(a1.multibody.MakeActuationMatrix())
    # Get the default A1 standing pose
    pose = a1.standing_pose()
    print(f"A1 standing pose{pose}")
    # Get a new standing pose using inverse kinematics
    pose2 = pose.copy()
    pose2[6] = pose[6]/2.
    pose2_ik, status = a1.standing_pose_ik(base_pose = pose2[0:7], guess = pose2.copy())
    print(f"IK Successful? {status}")
    print(f"Second A1 standing pose {pose2_ik}")
    #u, f = a1.static_controller(pose, verbose=True)
    print('Complete')
    #a1.configuration_sweep()
    # a1.print_frames()
    # pos = a1.standing_pose()
    # pos2 = pos.copy()
    # pos2[4] = 1
    # traj = PiecewisePolynomial.FirstOrderHold(np.linspace(0.,1.,101),np.linspace(pos, pos2, 101, axis=1))
    # a1.visualize(traj)
    # print(f"The configuration is {pos}")
    # a1.print_frames(pos)
    
