"""
Description of A1 robot

Updated January 14, 2020
Includes classes for creating A1 MultibodyPlant and TimesteppingMultibodyPlant as well as an A1Visualizer class
"""
# Library imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
from pydrake.all import PiecewisePolynomial, InverseKinematics, Solve, Body, PrismaticJoint, BallRpyJoint, SpatialInertia, UnitInertia, RevoluteJoint
# Project Imports
from pycito.utilities import FindResource, quat2rpy
from pycito.systems.timestepping import TimeSteppingMultibodyPlant
from pycito.systems.visualization import Visualizer
from pycito.systems.terrain import FlatTerrain
import pycito.decorators as deco
import pycito.utilities as utils

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

    def get_foot_position_in_world(self, context):
        world = self.multibody.world_frame()
        feet = []
        for pose, frame in zip(self.collision_poses, self.collision_frames):
            point = pose.translation().copy()
            feet.append(self.multibody.CalcPointsPositions(context, frame, point, world))
        return feet

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

    def plot_trajectories(self, xtraj=None, utraj=None, ftraj=None, jltraj=None, samples=None, show=False, savename=None):
        """ Plot all the trajectories for A1 """
        # Edit the save string
        if xtraj:
            self.plot_state_trajectory(xtraj, samples, show=False, savename=savename)
        if utraj:
            self.plot_control_trajectory(utraj, samples, show=False, savename=utils.append_filename(savename, '_control'))
        if ftraj:
            self.plot_force_trajectory(ftraj, samples, show=False, savename=utils.append_filename(savename, '_reactions'))
        if jltraj:
            self.plot_limit_trajectory(jltraj, samples, show=False, savename=utils.append_filename(savename, '_limits'))
        if show:
            plt.show()

    def plot_state_trajectory(self, xtraj, samples=None, show=True, savename=None):
        """ Plot the state trajectory for A1"""
        # Get the configuration and velocity trajectories as arrays
        t, x = utils.trajectoryToArray(xtraj, samples)
        nq = self.multibody.num_positions()
        q, v = np.split(x, [nq])
        # Get orientation from quaternion
        q[1:4] = quat2rpy(q[0:4,:])
        # Plot Base orientation and position
        self._plot_base_position(t, q[1:7,:], show=show, savename=utils.append_filename(savename, 'BaseConfiguration'))
        # Plot Base Velocity
        self._plot_base_velocity(t, v[0:6,:], show=show, savename=utils.append_filename(savename, 'BaseVelocity'))
        # Plot Joint Angles
        self._plot_joint_position(t, q[7:, :], show=show, savename=utils.append_filename(savename, 'JointAngles'))
        # Plot joint velocities
        self._plot_joint_velocity(t, v[6:,:], show=show, savename=utils.append_filename(savename, 'JointVelocity'))
        
    @deco.showable_fig
    @deco.saveable_fig
    def _plot_base_position(self, t, pos):
        """Make a plot of the position and orientation of the base"""
        fig, axs = plt.subplots(2,1)
        labels=[["Roll", "Pitch", "Yaw"],["X", "Y", "Z"]]
        ylabels = ["Orientation","Position"]
        for n in range(2):
            for k in range(3):
                axs[n].plot(t, pos[3*n + k,:], linewidth=1.5, label=labels[n][k])
            axs[n].set_ylabel(ylabels[n])
            axs[n].legend()
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("Base Configuration")
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def _plot_base_velocity(self, t, vel):
        """Plot COM orientation rate and translational velocity"""
        fig, axs = plt.subplots(2,1)
        labels=[["Roll", "Pitch", "Yaw"],["X", "Y", "Z"]]
        ylabels = ["Orientation","Position"]
        for n in range(2):
            for k in range(3):
                axs[n].plot(t, vel[3*n + k,:], linewidth=1.5, label=labels[n][k])
            axs[n].set_ylabel(ylabels[n] + " Rate")
            axs[n].legend()
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("COM Velocities")
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def _plot_joint_position(self, t, jpos):
        """Plot joint positions"""
        fig, axs = plt.subplots(3,1)
        # Loop over each joint angle
        legs = ["FR", "FL", "BR", "BL"]
        angles = ["Hip Roll", "Hip Pitch","Knee Pitch"]
        for n in range(3):
            # Loop over each leg
            for k in range(4):
                axs[n].plot(t, jpos[4*n+k,:], linewidth=1.5, label=legs[k])
            axs[n].set_ylabel(angles[n])
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("Joint Angles")
        axs[0].legend()
        return fig, axs   

    @deco.showable_fig
    @deco.saveable_fig
    def _plot_joint_velocity(self, t, jvel):
        """Plot the joint velocities"""
        fig, axs = plt.subplots(3,1)
        # Loop over each joint angle
        legs = ["FR", "FL", "BR", "BL"]
        angles = ["Hip Roll", "Hip Pitch","Knee Pitch"]
        for n in range(3):
            # Loop over each leg
            for k in range(4):
                axs[n].plot(t, jvel[4*n+k, :], linewidth=1.5, label=legs[k])
            axs[n].set_ylabel(angles[n] + " Rate")
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("Joint Rates")
        axs[0].legend() 
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_control_trajectory(self, utraj, samples=None):
        """Plot joint actuation torque trajectories"""
        # Get the knot points from the trajectory
        t, u = utils.trajectoryToArray(utraj, samples)
        # Plot the actuation torques, organized by joint angle
        fig, axs = plt.subplots(3,1)
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
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_force_trajectory(self, ftraj, samples=None):
        """ Plot reaction force trajectories"""
        t, f = utils.trajectoryToArray(ftraj, samples)
        f = self.resolve_forces(f)  
        fig, axs = plt.subplots(3,1)
        legs = ['FR', 'FL', 'BR', 'BL']
        labels = ['Normal', 'Friction-1', 'Friction-2']
        for k in range(3):
            for n in range(4):
                axs[k].plot(t, f[n + 4*k,:], linewidth=1.5, label=legs[n])
            axs[k].set_ylabel(labels[k])
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title('Reaction Forces')
        axs[0].legend()
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def plot_limit_trajectory(self, jltraj, samples=None):
        """
        Plot the joint limit torque trajectories
        """
        t, jl = utils.trajectoryToArray(jltraj, samples)
        jl = self.resolve_limit_forces(jl)
        leg = ['FR','FL','BR','BL']
        angle = ['Hip Roll','Hip Pitch','Knee Pitch']
        fig, axs = plt.subplots(3,1)
        for n in range(3):
            for k in range(4):
                axs[n].plot(t, jl[4*n + k,:], linewidth=1.5, label=leg[k])
            axs[n].set_ylabel(angle[n])
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title('Joint Limit Torques')
        axs[0].legend()
        return fig, axs

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
                
        # Sweep the COM translation
        for n in range(3):
            pos_ = pos.copy()
            pos_[n,:] = trans
            t_ = t + trajectory.end_time()
            trajectory.ConcatenateInTime(PiecewisePolynomial.FirstOrderHold(t_, pos_))

        # Sweep the joint positions
        for n in range(3,pos.shape[0]):
            pos_ = pos.copy()
            pos_[n,:] = angle
            t_ = t + trajectory.end_time()
            trajectory.ConcatenateInTime(PiecewisePolynomial.FirstOrderHold(t_, pos_))

        A1VirtualBase.visualize(trajectory)

    def standing_pose(self):
        # Get the default configuration vector
        context = self.multibody.CreateDefaultContext()
        pos = self.multibody.GetPositions(context)
        # Change the hip pitch to 45 degrees
        hip_pitch = [7, 10, 13, 16]
        pos[hip_pitch] = np.pi/4
        # Change the knee pitch to keep the foot under the hip
        knee_pitch = [8, 11, 14, 17]
        pos[knee_pitch] = -np.pi/2
        # Adjust the base height to make the normal distances 0
        self.multibody.SetPositions(context, pos)
        dist = self.GetNormalDistances(context)
        pos[2] = pos[2] - np.amin(dist)
        return pos

    def foot_pose_ik(self, base_pose, feet_position, guess=None):
        """
        Solves for the joint configuration that achieves the desired base and foot positions
        
        Arguments:
            base_pose: (6,) numpy array specifying base position as [xyz, rpy]
            feet_position: 4-iterable, each element a (3,) numpy array specifying foot positions for [FL, FR, RL, RR] in order
        Returns:    
            q: The solved configuration
            status: a boolean. True if the IK problem was solved successfully
        """
        # Create an IK problem
        IK = InverseKinematics(self.multibody, with_joint_limits=True)
        # Create context and set pose
        context = self.multibody.CreateDefaultContext()
        q_0 = np.zeros((self.multibody.num_positions(),))
        q_0[0:6] = base_pose
        self.multibody.SetPositions(context, q_0)
        # Constrain the foot positions in the world frame
        world = self.multibody.world_frame()
        for frame, fpose, wpose in zip(self.collision_frames, self.collision_poses, feet_position):
            point = fpose.translation().copy()
            IK.AddPositionConstraint(frame, point, world, wpose, wpose)
        # Set the base position as a constraint
        qvar = IK.q()
        prog = IK.prog()
        prog.AddLinearEqualityConstraint(Aeq = np.eye(6), beq = np.expand_dims(base_pose, axis=1), vars=qvar[0:6])
        # Set an initial guess
        if guess is None:
            guess = self.standing_pose()
        prog.SetInitialGuess(qvar, guess)
        # Solve the problem
        result = Solve(prog)
        return result.GetSolution(qvar), result.is_success()

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
        #TODO: FIX IK so that all points on the collision sphere are above the terrain. Current implementation only constrains the bottom of  the sphere, which is only valid if the legs are straight - ie knees locked
        world = self.multibody.world_frame()
        for pose, frame, radius in zip(self.collision_poses, self.collision_frames, self.collision_radius):
            point = pose.translation().copy()
            point_w = self.multibody.CalcPointsPositions(context, frame, point, world)
            point_w[-1] = radius
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

    def plot_state_trajectory(self, xtraj, samples=None, show=True, savename=None):
        """ Plot the state trajectory for A1"""
        # Get the configuration and velocity trajectories as arrays
        t, x = utils.trajectoryToArray(xtraj, samples)
        nq = self.multibody.num_positions()
        q, v = np.split(x, [nq])
        # Plot Base orientation and position
        self._plot_base_position(t, q[[3,4,5,0,1,2],:], show=show, savename=utils.append_filename(savename, 'BaseConfiguration'))
        # Plot Base Velocity
        self._plot_base_velocity(t, v[[3, 4, 5, 0, 1, 2],:], show=show, savename=utils.append_filename(savename, 'BaseVelocity'))
        # Plot Joint Angles
        self._plot_joint_position(t, q[6:, :], show=show, savename=utils.append_filename(savename, 'JointAngles'))
        # Plot joint velocities
        self._plot_joint_velocity(t, v[6:,:], show=show, savename=utils.append_filename(savename, 'JointVelocity'))

    @staticmethod
    def visualize(trajectory=None):
        vis = Visualizer("systems/A1/A1_description/urdf/a1_no_collision.urdf")
        # Add in virtual joints to represent the floating base
        zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
        # Create virtual, zero-mass
        xlink = vis.plant.AddRigidBody('xlink', vis.model_index, zeroinertia)
        ylink = vis.plant.AddRigidBody('ylink', vis.model_index, zeroinertia)
        zlink = vis.plant.AddRigidBody('zlink', vis.model_index, zeroinertia)
        # Create the translational and rotational joints
        xtrans = PrismaticJoint("xtranslation", vis.plant.world_frame(), xlink.body_frame(), [1., 0., 0.])
        ytrans = PrismaticJoint("ytranslation", xlink.body_frame(), ylink.body_frame(), [0., 1., 0.])
        ztrans = PrismaticJoint("ztranslation", ylink.body_frame(), zlink.body_frame(), [0., 0., 1.])
        rpyrotation = BallRpyJoint("baseorientation", zlink.body_frame(), vis.plant.GetBodyByName('base').body_frame())
        # Add the joints to the multibody plant
        vis.plant.AddJoint(xtrans)
        vis.plant.AddJoint(ytrans)
        vis.plant.AddJoint(ztrans)
        vis.plant.AddJoint(rpyrotation)
        vis.visualize_trajectory(xtraj=trajectory)

class PlanarA1(A1):
    def __init__(self, urdf_file="systems/A1/A1_description/urdf/a1_foot_collision.urdf", terrain=FlatTerrain()):
        super(PlanarA1, self).__init__(urdf_file, terrain)
        # Fix A1 so that it can only move in the xz plane
        zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
        # Create new links and joints
        xlink = self.multibody.AddRigidBody('xlink', self.model_index[0], zeroinertia)
        zlink = self.multibody.AddRigidBody('zlink', self.model_index[0], zeroinertia)
        xtrans = PrismaticJoint("xtranslation", self.multibody.world_frame(), xlink.body_frame(), [1., 0., 0.])
        ztrans = PrismaticJoint("ztranslation", xlink.body_frame(), zlink.body_frame(), [0., 0., 1.])
        yrotation = RevoluteJoint('yrotation', zlink.body_frame(), self.multibody.GetBodyByName('base').body_frame(), [0., 1., 0.])
        # Add the joints to the plant
        self.multibody.AddJoint(xtrans)
        self.multibody.AddJoint(ztrans)
        self.multibody.AddJoint(yrotation)

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
                
        # Sweep the COM translation
        for n in range(2):
            pos_ = pos.copy()
            pos_[n,:] = trans
            t_ = t + trajectory.end_time()
            trajectory.ConcatenateInTime(PiecewisePolynomial.FirstOrderHold(t_, pos_))

        # Sweep the joint positions
        for n in range(2,pos.shape[0]):
            pos_ = pos.copy()
            pos_[n,:] = angle
            t_ = t + trajectory.end_time()
            trajectory.ConcatenateInTime(PiecewisePolynomial.FirstOrderHold(t_, pos_))

        PlanarA1.visualize(trajectory)

    def standing_pose(self):
        # Get the default configuration vector
        context = self.multibody.CreateDefaultContext()
        pos = self.multibody.GetPositions(context)
        # Change the hip pitch to 45 degrees
        hip_pitch = [4, 7, 10, 13]
        pos[hip_pitch] = np.pi/4
        # Change the knee pitch to keep the foot under the hip
        knee_pitch = [5, 8, 11, 14]
        pos[knee_pitch] = -np.pi/2
        # Adjust the base height to make the normal distances 0
        self.multibody.SetPositions(context, pos)
        dist = self.GetNormalDistances(context)
        pos[1] = pos[1] - np.amin(dist)
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
        q_0[:3] = base_pose
        self.multibody.SetPositions(context, q_0)
        # Constrain the foot positions
        #TODO: FIX IK so that all points on the collision sphere are above the terrain. Current implementation only constrains the bottom of  the sphere, which is only valid if the legs are straight - ie knees locked
        world = self.multibody.world_frame()
        for pose, frame, radius in zip(self.collision_poses, self.collision_frames, self.collision_radius):
            point = pose.translation().copy()
            point_w = self.multibody.CalcPointsPositions(context, frame, point, world)
            point_w[-1] = radius
            IK.AddPositionConstraint(frame, point, world, point_w, point_w)
        # Set the base position as a constraint
        q_vars = IK.q()
        prog = IK.prog()
        prog.AddLinearEqualityConstraint(Aeq = np.eye(3), beq = np.expand_dims(base_pose, axis=1), vars=q_vars[:3])
        # Solve the problem
        prog = IK.prog()
        if guess is not None:
            # Set the initial guess
            prog.SetInitialGuess(q_vars, guess)
        result = Solve(prog)
        # Return the configuration vector
        return result.GetSolution(IK.q()), result.is_success()

    def foot_pose_ik(self, base_pose, feet_position, guess=None):
        """
        Solves for the joint configuration that achieves the desired base and foot positions
        
        Arguments:
            base_pose: (6,) numpy array specifying base position as [xyz, rpy]
            feet_position: 4-iterable, each element a (3,) numpy array specifying foot positions for [FL, FR, RL, RR] in order
        Returns:    
            q: The solved configuration
            status: a boolean. True if the IK problem was solved successfully
        """
        # Create an IK problem
        IK = InverseKinematics(self.multibody, with_joint_limits=True)
        # Create context and set pose
        context = self.multibody.CreateDefaultContext()
        q_0 = np.zeros((self.multibody.num_positions(),))
        q_0[:3] = base_pose
        self.multibody.SetPositions(context, q_0)
        # Constrain the foot positions in the world frame
        world = self.multibody.world_frame()
        for frame, fpose, wpose in zip(self.collision_frames, self.collision_poses, feet_position):
            point = fpose.translation().copy()
            IK.AddPositionConstraint(frame, point, world, wpose, wpose)
        # Set the base position as a constraint
        qvar = IK.q()
        prog = IK.prog()
        prog.AddLinearEqualityConstraint(Aeq = np.eye(3), beq = np.expand_dims(base_pose, axis=1), vars=qvar[:3])
        # Set an initial guess
        if guess is None:
            guess = self.standing_pose()
        prog.SetInitialGuess(qvar, guess)
        # Solve the problem
        result = Solve(prog)
        return result.GetSolution(qvar), result.is_success()

    def plot_state_trajectory(self, xtraj, samples=None, show=True, savename=None):
        """ Plot the state trajectory for A1"""
        # Get the configuration and velocity trajectories as arrays
        t, x = utils.trajectoryToArray(xtraj, samples)
        nq = self.multibody.num_positions()
        q, v = np.split(x, [nq])
        # Plot Base orientation and position
        self._plot_base_position(t, q[:3, :], show=show, savename=utils.append_filename(savename, 'BaseConfiguration'))
        # Plot Base Velocity
        self._plot_base_velocity(t, v[:3, :], show=show, savename=utils.append_filename(savename, 'BaseVelocity'))
        # Plot Joint Angles
        self._plot_joint_position(t, q[3:, :], show=show, savename=utils.append_filename(savename, 'JointAngles'))
        # Plot joint velocities
        self._plot_joint_velocity(t, v[3:, :], show=show, savename=utils.append_filename(savename, 'JointVelocity'))
        
    @deco.showable_fig
    @deco.saveable_fig
    def _plot_base_position(self, t, pos):
        """Make a plot of the position and orientation of the base"""
        fig, axs = plt.subplots(3,1)
        labels=["Horizontal (m)", "Vertical (m)", "Rotation (rad)"]
        for n in range(3):
            axs[n].plot(t, pos[n,:], linewidth=1.5)
            axs[n].set_ylabel(labels[n])
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("Base Configuration")
        return fig, axs

    @deco.showable_fig
    @deco.saveable_fig
    def _plot_base_velocity(self, t, vel):
        """Plot COM orientation rate and translational velocity"""
        fig, axs = plt.subplots(3, 1)
        labels=["Horizontal (m/s)", "Vertical (m/s)", "Rotation (rad/s)"]
        for n in range(3):
            axs[n].plot(t, vel[n,:], linewidth=1.5)
            axs[n].set_ylabel(labels[n])
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title("Base Velocities")
        return fig, axs

    @staticmethod
    def visualize(trajectory=None):
        vis = Visualizer("systems/A1/A1_description/urdf/a1_no_collision.urdf")
        # Add in virtual joints to represent the floating base
        zeroinertia = SpatialInertia(0, np.zeros((3,)), UnitInertia(0., 0., 0.))
        # Create new links and joints
        xlink = vis.plant.AddRigidBody('xlink', vis.model_index, zeroinertia)
        zlink = vis.plant.AddRigidBody('zlink', vis.model_index, zeroinertia)
        xtrans = PrismaticJoint("xtranslation", vis.plant.world_frame(), xlink.body_frame(), [1., 0., 0.])
        ztrans = PrismaticJoint("ztranslation", xlink.body_frame(), zlink.body_frame(), [0., 0., 1.])
        yrotation = RevoluteJoint('yrotation', zlink.body_frame(), vis.plant.GetBodyByName('base').body_frame(), [0., 1., 0.])
        # Add the joints to the multibody plant
        vis.plant.AddJoint(xtrans)
        vis.plant.AddJoint(ztrans)
        vis.plant.AddJoint(yrotation)
        vis.visualize_trajectory(xtraj=trajectory)

def describe_a1():
    a1 = A1()
    a1.Finalize()
    print(f"A1 effort limits {a1.get_actuator_limits()}")
    qmin, qmax = a1.get_joint_limits()
    print(f"A1 has lower joint limits {qmin} and upper joint limits {qmax}")
    print(f"A1 has actuation matrix:")
    print(a1.multibody.MakeActuationMatrix())
    for frame, pose in zip(a1.collision_frames, a1.collision_poses):
        print(f"A1 has collision frame with name {frame.name()}")

def example_ik():
    a1 = A1()
    a1.Finalize()
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

def example_pose():
    a1 = A1VirtualBase()
    a1.Finalize()
    pos = a1.standing_pose()
    pos2 = pos.copy()
    pos2[0] = 1
    traj = PiecewisePolynomial.FirstOrderHold(np.linspace(0.,1.,101),np.linspace(pos, pos2, 101, axis=1))
    a1.visualize(traj)
    print(f"The configuration is {pos}")
    context = a1.multibody.CreateDefaultContext()
    a1.multibody.SetPositions(context,pos)
    print(f"The normal distances are {a1.GetNormalDistances(context)}")
    
def run_configuration_sweep():
    a1 = PlanarA1()
    a1.Finalize()
    a1.configuration_sweep()

if __name__ == "__main__":
    #describe_a1()
    run_configuration_sweep()
    #example_pose()