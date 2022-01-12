import numpy as np
from systems.A1.a1 import A1VirtualBase
from pydrake.all import PiecewisePolynomial
import matplotlib.pyplot as plt
from math import ceil
import utilities as utils
import os

#TODO: Fix problem with static force controller
#TODO: Fix problem with beginning and ending half-cycles. Makes the gait look like a march
#TODO: Fix problems with nonzero foot heights

def reflect_trajectory(traj):
    start_t = traj.start_time()
    end_t = traj.end_time()
    traj.ReverseTime()
    traj.shiftRight(start_t + end_t)
    return traj

class GaitParameters():
    def __init__(self, step_length=0.2, step_height=0.1, swing_phase=0.4, cycle_duration=1.0):
        self.step_length = step_length
        self.step_height = step_height
        self.swing_phase = swing_phase
        self.cycle_duration = cycle_duration

class A1WarmstartGenerator(A1VirtualBase):
    def __init__(self):
        super(A1WarmstartGenerator, self).__init__()
        self.Finalize()

    def _linear_state_interpolation(self, x0, xf, N):
        x = np.zeros((x0.shape[0], N))
        for n in range(x0.shape[0]):
            x[n,:] = np.linspace(x0[n], xf[n], N)
        return x

    def linearWarmstart(self, distance=1, N=101):
        # Get the standing pose
        q0_ = self.standing_pose()
        q0, status = self.standing_pose_ik(q0_[0:6], guess=q0_)
        # Make the state vector
        x0 = np.zeros((2*q0.shape[0]))
        x0[:q0.shape[0]] = q0
        xf = x0.copy()
        xf[0] = distance
        # Interpolate between initial and goal positions
        return self._linear_state_interpolation(x0, xf, N)
        
    def linearLiftedWarmstart(self, distance=1, footheight=0.1, N=101):
        # Set the standing pose
        q0_ = self.standing_pose()
        q0, status = self.standing_pose_ik(q0_[0:6], guess=q0_)
        context = self.multibody.CreateDefaultContext()
        self.multibody.SetPositions(context, q0)
        # Get the foot positions in world coordinates
        world = self.multibody.world_frame()
        feet_positions = []
        for pose, frame in zip(self.collision_poses, self.collision_frames):
            point = pose.translation().copy()
            point_w = self.multibody.CalcPointsPositions(context, frame, point, world)
            point_w[-1] = footheight
            feet_positions.append(point_w)
        # Solve the general IK problem
        q, status = self.foot_pose_ik(base_pose = q0[:6], feet_position=feet_positions, guess=q0)
        # Make the state vector
        x0 = np.zeros((2*q.shape[0]))
        x0[:q.shape[0]] = q
        xf = x0.copy()
        xf[0] = distance
        # Interpolate between initial and goal positions
        return self._linear_state_interpolation(x0, xf, N)

class A1GaitGenerator(A1VirtualBase):
    def __init__(self):
        super(A1GaitGenerator, self).__init__()
        self.Finalize()

    @staticmethod
    def make_foot_swing_trajectory(x0 = np.zeros((3,)), gait=GaitParameters()):
        x0 = np.squeeze(x0)
        t = np.array([0, 0.5, 1.])
        x = np.array([[x0[0], gait.step_length/2, gait.step_length/2],
                    [x0[1], 0., 0.],
                    [x0[2], gait.step_height, -gait.step_height]])
        x = np.cumsum(x, axis=1)
        return PiecewisePolynomial.LagrangeInterpolatingPolynomial(t, x)

    @staticmethod
    def make_foot_stance_trajectory(x0 = np.zeros((3,))):
        x0 = np.squeeze(x0)
        t = np.array([0., 1.])
        x = np.column_stack([x0, x0])
        return PiecewisePolynomial.LagrangeInterpolatingPolynomial(t, x)

    @staticmethod
    def make_leading_leg_cycle(x0 = np.zeros((3,)), gait=GaitParameters(), reversed=False):
        swing_traj = A1GaitGenerator.make_foot_swing_trajectory(x0, gait)
        xf = swing_traj.vector_values([swing_traj.end_time()])
        stance_traj = A1GaitGenerator.make_foot_stance_trajectory(np.squeeze(xf))
        swing_traj.ScaleTime(gait.swing_phase)
        ds = 1 - 2 * gait.swing_phase   #Double support duration
        if reversed:
            if ds > 0:
                # With double support
                lag = ds/2
                stance_lag = A1GaitGenerator.make_foot_stance_trajectory(xf)
                stance_traj = A1GaitGenerator.make_foot_stance_trajectory(x0)
                stance_lag.ScaleTime(lag)
                stance_lag.shiftRight(swing_traj.end_time())
                swing_traj.ConcatenateInTime(stance_lag)
                stance_traj.ScaleTime(1 - swing_traj.end_time())
                swing_traj.shiftRight(stance_traj.end_time())
            else:
                # No double support
                stance_traj.ScaleTime(1-gait.swing_phase)
                swing_traj.shiftRight(1-gait.swing_phase)
            # Combine swing and stance trajectories
            stance_traj.ConcatenateInTime(swing_traj)
            stance_traj.ScaleTime(gait.cycle_duration)
            return stance_traj
        else:
            if ds > 0:
                lag = ds/2  
                stance_lag = A1GaitGenerator.make_foot_stance_trajectory(x0)
                stance_lag.ScaleTime(lag)
                swing_traj.shiftRight(stance_lag.end_time())
                stance_lag.ConcatenateInTime(swing_traj)
                stance_traj.ScaleTime(1 - stance_lag.end_time())
                stance_traj.shiftRight(stance_lag.end_time())
                stance_lag.ConcatenateInTime(stance_traj)
                stance_lag.ScaleTime(gait.cycle_duration)
                return stance_lag
            else:
                # No double support phase
                stance_traj.ScaleTime(1-gait.swing_phase)
                stance_traj.shiftRight(gait.swing_phase)
                swing_traj.ConcatenateInTime(stance_traj)
                swing_traj.ScaleTime(gait.cycle_duration)
                return swing_traj

    @staticmethod
    def make_trailing_leg_cycle(x0 = np.zeros((3,)), gait=GaitParameters()):
        swing_traj = A1GaitGenerator.make_foot_swing_trajectory(x0, gait)
        stance_traj = A1GaitGenerator.make_foot_stance_trajectory(x0)
        swing_traj.ScaleTime(gait.swing_phase)
        stance_traj.ScaleTime(1-gait.swing_phase)
        swing_traj.shiftRight(1-gait.swing_phase)
        stance_traj.ConcatenateInTime(swing_traj)
        stance_traj.ScaleTime(gait.cycle_duration)
        return stance_traj

    @staticmethod
    def make_a1_gait_cycle(FL, FR, BL, BR, gait=GaitParameters()):
        fl_traj = A1GaitGenerator.make_trailing_leg_cycle(FL, gait)
        fr_traj = A1GaitGenerator.make_leading_leg_cycle(FR, gait)
        bl_traj = A1GaitGenerator.make_leading_leg_cycle(BL, gait)
        br_traj = A1GaitGenerator.make_trailing_leg_cycle(BR, gait)
        return fl_traj, fr_traj, bl_traj, br_traj

    @staticmethod
    def make_leading_half_cycle(x0 = np.zeros((3,)), gait=GaitParameters(), reversed=False):
        halfGait = GaitParameters(gait.step_length/2, gait.step_height, gait.swing_phase, gait.cycle_duration/2)    
        return A1GaitGenerator.make_leading_leg_cycle(x0, halfGait, reversed)

    @staticmethod
    def make_trailing_half_cycle(x0 = np.zeros((3,)), gait=GaitParameters()):
        halfGait = GaitParameters(0., 0., gait.swing_phase, gait.cycle_duration/2)
        return A1GaitGenerator.make_trailing_leg_cycle(x0, halfGait)

    @staticmethod
    def make_a1_half_step(FL, FR, BL, BR, gait=GaitParameters(), reversed=False):
        if reversed:
            fl_traj = A1GaitGenerator.make_leading_half_cycle(FL, gait, reversed)
            fr_traj = A1GaitGenerator.make_trailing_half_cycle(FR, gait)
            bl_traj = A1GaitGenerator.make_trailing_half_cycle(BL, gait)
            br_traj = A1GaitGenerator.make_leading_half_cycle(BR, gait, reversed)
        else:
            fl_traj = A1GaitGenerator.make_trailing_half_cycle(FL, gait)
            fr_traj = A1GaitGenerator.make_leading_half_cycle(FR, gait)
            bl_traj = A1GaitGenerator.make_leading_half_cycle(BL, gait)
            br_traj = A1GaitGenerator.make_trailing_half_cycle(BR, gait)
        return fl_traj, fr_traj, bl_traj, br_traj

    @staticmethod
    def make_a1_gait(feetpose, numsteps = 4, gait=GaitParameters()):
        #TODO: add in first and last gait cycles
        FL, FR, BL, BR = feetpose
        # First gait cycle
        fl, fr, bl, br = A1GaitGenerator.make_a1_half_step(FL, FR, BL, BR, gait, reversed=True)
        for _ in range(1, numsteps):
            fl_, fr_, bl_, br_ = A1GaitGenerator.make_a1_gait_cycle(fl.vector_values([fl.end_time()]), 
                                                    fr.vector_values([fr.end_time()]), 
                                                    bl.vector_values([bl.end_time()]),
                                                    br.vector_values([br.end_time()]),
                                                    gait)
            fl_.shiftRight(fl.end_time())
            fr_.shiftRight(fr.end_time())
            bl_.shiftRight(bl.end_time())
            br_.shiftRight(br.end_time())
            fl.ConcatenateInTime(fl_)
            fr.ConcatenateInTime(fr_)
            bl.ConcatenateInTime(bl_)
            br.ConcatenateInTime(br_)
        # Last gait cycle
        fl_, fr_, bl_, br_ = A1GaitGenerator.make_a1_half_step(
            fl.vector_values([fl.end_time()]),
            fr.vector_values([fr.end_time()]),
            bl.vector_values([bl.end_time()]),
            br.vector_values([br.end_time()]),
            gait
        )
        # Concatenate
        fl_.shiftRight(fl.end_time())
        fr_.shiftRight(fr.end_time())
        bl_.shiftRight(bl.end_time())
        br_.shiftRight(br.end_time())
        fl.ConcatenateInTime(fl_)
        fr.ConcatenateInTime(fr_)
        bl.ConcatenateInTime(bl_)
        br.ConcatenateInTime(br_)
        return fl, fr, bl, br

    @staticmethod
    def make_a1_base_trajectory(q0, feet_traj):
        total_time = feet_traj[0].end_time()
        base_0 = q0[:6]
        breaks = feet_traj[0].get_segment_times()
        travel = np.row_stack([foot.vector_values(breaks)[0,:] for foot in feet_traj])
        offset = np.min(travel[:, 0])
        travel = travel - offset
        base_travel = np.average(travel, axis=0)
        base_travel += offset
        N = base_travel.shape[0]
        travel += offset
        base = np.repeat(np.expand_dims(base_0, axis=1), N, axis=1)
        base[0,:] = base_travel
        return PiecewisePolynomial.FirstOrderHold(breaks, base)

    def make_gait(self, distance = 1, gait=GaitParameters()):
        # Calculate the number of gait cycles necessary
        ncycles = int(ceil(distance / gait.step_length))   
        steady_dist = (ncycles-1)*gait.step_length
        total_dist = steady_dist + gait.step_length/2
        gait.step_length *= distance/total_dist
        # Start from the static standing pose
        q0_ = self.standing_pose()
        q0, _ = self.standing_pose_ik(base_pose=q0_[0:6], guess=q0_)
        # Get the foot positions
        context = self.multibody.CreateDefaultContext()
        self.multibody.SetPositions(context, q0)
        world = self.multibody.world_frame()
        foot_point = []
        for pose, frame, radius in zip(self.collision_poses, self.collision_frames, self.collision_radius):
            point = pose.translation().copy()
            wpoint = self.multibody.CalcPointsPositions(context, frame, point, world)
            wpoint = np.squeeze(wpoint)
            #wpoint[-1] -= radius
            foot_point.append(wpoint)
        # Calculate the base trajectory
        feet = self.make_a1_gait(foot_point, ncycles, gait)
        base = self.make_a1_base_trajectory(q0, feet)
        return feet, base

    def make_configuration_profile(self, feet_traj, base_traj, sampling=1000):
        """
        Run IK To get a configuration profile from the feet and base trajectories
        """
        q = np.zeros((self.multibody.num_positions(), sampling))
        t = np.linspace(base_traj.start_time(), base_traj.end_time(), sampling)
        base = base_traj.vector_values(t)
        feet = [foot_traj.vector_values(t) for foot_traj in feet_traj]
        q_ = self.standing_pose()
        feet_pos = [foot[:, 0] for foot in feet]
        q[:, 0], status = self.foot_pose_ik(base[:, 0], feet_pos, guess=q_)
        if not status:
            print(f"Foot position IK failed at index 0")
        for n in range(1, sampling):
            feet_pos = [foot[:, n] for foot in feet]
            q[:, n], status = self.foot_pose_ik(base[:, n], feet_pos, q[:, n-1])
            if not status:
                print(f"Foot position IK failed at index {n}")
        return q

    def make_force_profile(self, qtraj):
        """Return joint torques and reaction forces to make configuration trajectory static"""
        # Assume static, and calculate normal forces and joint torques necessary.    
        u = np.zeros((self.multibody.num_actuators(), qtraj.shape[1]))
        fN = np.zeros((4, qtraj.shape[1]))
        for n in range(qtraj.shape[1]):
            u[:, n], fN[:, n] = self.static_controller(qtraj[:, n])
        return u, fN

    def make_a1_walking_warmstart(self, distance=1, gait=GaitParameters(), sampling=1000):
        feet, base = self.make_gait(distance, gait)
        q = self.make_configuration_profile(feet,  base, sampling)
        u, fN = self.make_force_profile(q)
        return q, u, fN

def trajectory_example():
    print('Making trajectory')
    xtraj = A1GaitGenerator.make_foot_swing_trajectory()
    print('Getting values')
    t = np.linspace(0, 1, 1000)
    x = xtraj.vector_values(t)
    print('Creating plots')
    fig, axs = plt.subplots(3,1)
    labels = ['Forward (m)', 'Lateral (m)', 'Vertical (m)']
    for n in range(3):
        axs[n].plot(t, x[n,:], linewidth=1.5)
        axs[n].set_ylabel(labels[n])
    axs[-1].set_xlabel('Time (s)')
    print('plotting')
    plt.show()

def warmstart_examples():
    generator = A1WarmstartGenerator()
    t = np.linspace(0, 1, 101)  
    print('Generating linear warmstart')
    xlinear = generator.linearWarmstart(distance=1, N = 101)
    xlinear = PiecewisePolynomial.FirstOrderHold(t, xlinear)
    generator.visualize(xlinear)
    print('Generating lifted warmstart')
    xlifted = generator.linearLiftedWarmstart(distance = 1, footheight = 0.1, N = 101)
    xlifted = PiecewisePolynomial.FirstOrderHold(t, xlifted)
    generator.visualize(xlifted)
    print('Generating gait cycle warmstart')
    
def gait_example():
    generator = A1GaitGenerator()
    q, u, fN = generator.make_a1_walking_warmstart(distance=1, sampling=101)
    t = np.linspace(0, 1, 101)
    x = np.zeros((2*q.shape[0], q.shape[1]))
    x[0:q.shape[0]] = q
    xtraj = PiecewisePolynomial.FirstOrderHold(t, x)
    print('Visualizing gait')
    generator.visualize(xtraj)
    # Plot the controls and reaction forces
    f = np.zeros((2*generator.num_contacts() + generator.num_friction(), 101))
    f[:4, :] = fN
    utraj = PiecewisePolynomial.ZeroOrderHold(t, u)
    ftraj = PiecewisePolynomial.ZeroOrderHold(t, f)
    generator.plot_state_trajectory(xtraj, show=False)
    generator.plot_control_trajectory(utraj, show=False)
    generator.plot_force_trajectory(ftraj, show=False)
    plt.show()

def debug_gait():
    generator = A1GaitGenerator()
    feet, base = generator.make_gait(distance=1)
    fig, axs = plt.subplots(3,1)
    t = np.linspace(0, base.end_time(), 101)
    base = base.vector_values(t)
    feet = [foot.vector_values(t) for foot in feet]
    labels = ['base', 'FL','FR','BL','BR']
    ylabels = ['X','Y','Z']
    feet.insert(0, base)
    for n in range(3):
        for k in range(5):
            axs[n].plot(t, feet[k][n,:], linewidth=1.5, label=labels[k])
        axs[n].set_ylabel(ylabels[n])
    axs[-1].set_xlabel('Time (s)')
    axs[0].legend()
    plt.show()

def make_warmstarts():
    N = 51
    t = np.linspace(0, 1, N)
    generator1 = A1WarmstartGenerator()
    print('Generating Lifted Warmstart')
    xlifted = generator1.linearLiftedWarmstart(distance=1, footheight=0.1, N=N)
    u = np.zeros((generator1.multibody.num_actuators(), N))
    f = np.zeros((2*generator1.num_contacts()+generator1.num_friction(), N))
    warmstart = {'time': t,
                'state': xlifted,
                'control': u,
                'force': f,
                'jointlimit': np.zeros((xlifted.shape[0] - 12, N))}
    utils.save(os.path.join('data','a1','warmstarts',f'liftedlinear_{N}.pkl'), warmstart)
    print('Generating Walking Warmstart')
    generator2 = A1GaitGenerator()
    q, u, fN = generator2.make_a1_walking_warmstart(distance=1, sampling=N)
    # Expand the state and force trajectories
    x = np.zeros((2*q.shape[0], q.shape[1]))
    x[0:q.shape[0]] = q
    f = np.zeros((2*generator2.num_contacts() + generator2.num_friction(), N))
    f[:4, :] = fN
    warmstart2 = {'time': t,
                'state': x,
                'control': u,
                'force': f,
                'jointlimit': np.zeros((2*(q.shape[0]-6), N))}
    utils.save(os.path.join('data','a1','warmstarts',f'staticwalking_{N}.pkl'), warmstart2)

if __name__ == "__main__":
    debug_gait()
    #gait_example()
    #make_warmstarts()