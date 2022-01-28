"""
Gait trajectory optimization for A1.

Runs trajectory optimization for A1 with a nonlinear foot-tracking cost

Luke Drnach
January 12, 2022
"""
import numpy as np
import os
import matplotlib.pyplot as plt

from pydrake.all import PiecewisePolynomial as PP

import pycito.utilities as utils
import a1trajopttools as opttools

class A1FootTrackingCost():
    def __init__(self, a1, weight, traj):
        # Save the plant
        self.a1_f = a1
        self.a1_ad = a1.toAutoDiffXd()     
        self.context_f = self.a1_f.multibody.CreateDefaultContext()
        self.context_ad = self.a1_ad.multibody.CreateDefaultContext()
        # Save the cost terms
        self.cost_weight = weight
        self.foot_traj = traj

    def addToProgram(self, prog, qvars):
        """Add the cost to an existing mathematical program"""
        prog.AddCost(self.eval_cost, vars=qvars.flatten(), description='FootTrackingCost')
        self._q_shape = qvars.shape
        return prog

    def eval_cost(self, qvars):
        """
        Evaluate the tracking cost
        """
        qvars = np.reshape(qvars, self._q_shape)
        cost = 0
        a1, context = self._autodiff_or_float(qvars)
        for k, q in enumerate(qvars.T):
            a1.multibody.SetPositions(context, q)
            foot_q = a1.get_foot_position_in_world(context)
            for foot, foot_ref in zip(foot_q, self.foot_traj):
                cost += self.cost_weight * (foot_ref[:, k] - foot[:,0]).dot(foot_ref[:, k] - foot[:,0])
        return cost

    def _autodiff_or_float(self, z):
        if z.dtype == "float":
            return (self.a1_f, self.context_f)
        else:
            return (self.a1_ad, self.context_ad)

class GaitType():
    def __init__(self, stride_length = 0.2, step_height=0.1, swing_phase=0.8, duration=0.5):
        self.stride_length = stride_length
        self.step_height = step_height
        self.swing_phase = swing_phase
        self.duration = duration
        self.reversed = False
        self.generator = GaitType.make_ellipse_trajectory

    def use_ellipse(self):
        self.generator = GaitType.make_ellipse_trajectory

    def use_quadratic(self):
        self.generator = GaitType.make_quadratic_trajectory

    def use_quartic(self):
        self.generator = GaitType.make_quartic_trajectory

    @staticmethod
    def make_ellipse_trajectory(xstart, xgoal, zpeak, samples):
        d = 0.5*(xgoal - xstart)
        x = np.linspace(xstart, xgoal, samples)
        r = 1 - (x - (xstart + d))**2  / (d**2)
        r[r < 0] = 0
        z = zpeak * np.sqrt(r)
        return x, z

    @staticmethod
    def make_quadratic_trajectory(xstart, xgoal, zpeak, samples):
        x, z = GaitType.make_ellipse_trajectory(xstart, xgoal, 1., samples)
        z = zpeak * (z**2)
        return x, z

    @staticmethod
    def make_quartic_trajectory(xstart, xgoal, zpeak, samples):
        x, z = GaitType.make_ellipse_trajectory(xstart, xgoal, 1., samples)
        z = zpeak * (z**4)
        return x, z

class GaitGenerator():
    def __init__(self, a1):
        self.a1 = a1

    def make_foot_stride_trajectory(self, q0, gait=GaitType(), samples=101):
        # Get the starting feet positions
        context = self.a1.multibody.CreateDefaultContext()
        self.a1.multibody.SetPositions(context, q0)
        feet = self.a1.get_foot_position_in_world(context)  #FOOT ORDERING: FR, FL, BR, BL
        # Create a single swing trajectory for FR and BL
        feet = [np.repeat(foot, repeats = samples, axis=1) for foot in feet]
        if gait.reversed:
            # Make trajectories for FR, BL (index 0, 3)
            swing_idx = [0, 3]        
        else:  
            # Make trajectories for FL, BR (index 1, 2)
            swing_idx = [1, 2]

        for idx in swing_idx:
            xSwing, zSwing = gait.generator(feet[idx][0,:], feet[idx][0,:] + gait.stride_length, gait.step_height, int(samples * gait.swing_phase))
            stance = (samples - int(gait.swing_phase * samples))//2
            feet[idx][0, stance:stance + xSwing.shape[0]] = xSwing[:, 0]
            feet[idx][0, stance+xSwing.shape[0]:] = xSwing[-1, 0]
            feet[idx][2, stance:stance + zSwing.shape[0]] = feet[idx][2,0] + zSwing[:, 0]
        return feet

    @staticmethod
    def make_base_trajectory(q0, feet_traj):
        base_0 = q0[:6]
        travel = np.row_stack([foot[0,:] for foot in feet_traj])
        offset = np.min(travel[:, 0])
        travel = travel - offset
        base_travel = np.average(travel, axis=0)
        base_travel += offset
        N = base_travel.shape[0]
        travel += offset
        base = np.repeat(np.expand_dims(base_0, axis=1), N, axis=1)
        base[0,:] = base_travel
        return base

    def make_configuration_profile(self, feet, base):
        """
        Run IK To get a configuration profile from the feet and base trajectories
        """
        q = np.zeros((self.a1.multibody.num_positions(), base.shape[1]))
        q_ = self.a1.standing_pose()
        feet_pos = [foot[:, 0] for foot in feet]
        q[:, 0], status = self.a1.foot_pose_ik(base[:, 0], feet_pos, guess=q_)
        if not status:
            print(f"Foot position IK failed at index 0")
        for n in range(1, base.shape[1]):
            feet_pos = [foot[:, n] for foot in feet]
            q[:, n], status = self.a1.foot_pose_ik(base[:, n], feet_pos, q[:, n-1])
            if not status:
                print(f"Foot position IK failed at index {n}")
        return q

def solve_forces(a1, qtraj):
    u = np.zeros((a1.multibody.num_actuators(), qtraj.shape[1]))
    fN = np.zeros((4, qtraj.shape[1]))
    for n in range(qtraj.shape[1]):
        u[:, n], fN[:, n] = a1.static_controller(qtraj[:, n])
    return u, fN

def make_a1_step_trajectories(a1):
    generator = GaitGenerator(a1)
    #Starting configuration
    q0 = a1.standing_pose()
    q0, _ = a1.standing_pose_ik(base_pose = q0[:6], guess=q0)
    # Make the gait types
    gaits = [GaitType(stride_length=0.1, duration = 0.25), 
            GaitType(stride_length=0.2, duration=0.5),
            GaitType(stride_length=0.2, duration=0.5),
            GaitType(stride_length=0.1, duration = 0.25)]
    gaits[1].reversed = True
    gaits[3].reversed = True
    for gait in gaits:
        gait.use_quartic()
    samples = [26, 51, 51, 26]
    # Make the gait trajectories
    step = []
    base = []
    q = []
    for gait, sample in zip(gaits, samples):
        step.append(generator.make_foot_stride_trajectory(q0, gait, sample))
        base.append(generator.make_base_trajectory(q0, step[-1]))
        q.append(generator.make_configuration_profile(step[-1], base[-1]))
        q0 = q[-1][:, -1]

    return step, base, q

def visualize_foot_trajectory(a1):
    feet_, base, q = make_a1_step_trajectories(a1)
    feet = []
    for k in range(4):
        feet.append(np.concatenate([foot[k] for foot in feet_ ], axis=1))
    #feet = np.concatenate(feet, axis=0)
    base = np.concatenate(base, axis=1)
    q = np.concatenate(q, axis=1)
    t = np.linspace(0, 1, q.shape[1])
    x = np.concatenate([q, np.zeros_like(q)], axis=0)
    xtraj = PP.FirstOrderHold(t, x)
    a1.visualize(xtraj)
    # Plot the data
    _, axs = plt.subplots(3,1)
    labels = ['FR','FL','BR','BL']  
    axislabels = ['X','Y','Z']
    for n in range(3):
        for k, foot in enumerate(feet):
            axs[n].plot(t, foot[n,:], linewidth=1.5, label=labels[k])
        axs[n].set_ylabel(axislabels[n])
    axs[-1].set_xlabel('Time (s)')
    axs[0].legend()
    plt.show()

def check_foot_tracking_cost(a1):
    foot, _, q = make_a1_step_trajectories(a1)
    footCost = A1FootTrackingCost(a1, weight=1, traj = foot[0])
    cost = footCost.eval_cost(q[0])
    print(f'FootCost = {cost}')

def add_base_tracking(trajopt, qref, weight):
    base_vars = trajopt.x[:6,:].flatten()
    base_vals = qref[:6,:].flatten()
    Q = weight * np.eye(base_vars.shape[0])
    trajopt.prog.AddQuadraticErrorCost(Q, base_vals, base_vars).evaluator().set_description('BaseTrackingCost')
    return trajopt

def optimize_foot_tracking_gait(a1, foot_ref, base_ref, duration, warmstart, savedir, options=None):
    """Generically solve the trajopt with foot and base tracking costs"""
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    N = warmstart['state'].shape[1]
    nQ = a1.multibody.num_positions()
    # Setup the trajectory optimization
    trajopt = opttools.make_a1_trajopt_linearcost(a1, N, [duration, duration])
    # Add boundary constraints
    trajopt = opttools.add_boundary_constraints(trajopt, warmstart['state'][:, 0], warmstart['state'][:, -1])
    # Set the initial guess
    trajopt.set_initial_guess(xtraj=warmstart['state'], utraj=warmstart['control'], ltraj=warmstart['force'], jltraj=warmstart['jointlimit'],straj=warmstart['slacks'])
    # Require foot tracking                           
    foot_cost = A1FootTrackingCost(a1, weight=1e4, traj=foot_ref)
    trajopt.prog = foot_cost.addToProgram(trajopt.prog, qvars=trajopt.x[:nQ,:])
    # Require base tracking
    trajopt = add_base_tracking(trajopt, base_ref, weight=1e2)
    # Set a small cost on control
    trajopt = opttools.add_control_cost(trajopt, weight=1e-2)
    # Add small cost on force
    trajopt = opttools.add_force_cost(trajopt, weight=1e-2)
    # Add small cost on control difference
    trajopt = opttools.add_control_difference_cost(trajopt, weight=1e-2)
    # Add small cost on force difference
    trajopt = opttools.add_force_difference_cost(trajopt, weight=1e-2)
    # Update the solver options
    if options:
        trajopt.setSolverOptions(options)
    # Solve the problem using different complementarity cost weights
    weights = [1, 1e1, 1e2, 1e3]
    opttools.progressive_solve(trajopt, weights, savedir)

def main_step_optimization():
    savedir = os.path.join('examples','a1','foot_tracking_gait')
    parts = ['start_step','first_step','second_step','last_step']
    a1 = opttools.make_a1()
    feet, _, qtraj = make_a1_step_trajectories(a1)
    durations = [0.25, 0.5, 0.5, 0.25]
    warmstart = {'state': None,
                'control': None,
                'force': None,
                'jointlimit': None,
                'slacks': None}
    for q, foot, filepart, duration in zip(qtraj, feet, parts, durations):
        warmstart['control'], fN = solve_forces(a1, q)
        warmstart['force'] = np.zeros((2*a1.num_contacts() + a1.num_friction(), fN.shape[1]))
        warmstart['force'][:fN.shape[0], :] = fN
        warmstart['state'] = np.concatenate([q, np.zeros((a1.multibody.num_velocities(), q.shape[1]))], axis=0)
        optimize_foot_tracking_gait(a1, foot, q[:6, :], duration, warmstart, os.path.join(savedir, filepart))

""" For full gait optimization """
def getdatakeys():
    return ['state','control','force','jointlimit','slacks']

def load_full_gait():
    sources = ['start_step','first_step','second_step','last_step']
    datakeys = getdatakeys()
    fullgait = utils.load(os.path.join('examples','a1','foot_tracking_gait', sources[0],'weight_1e+03','trajoptresults.pkl'))
    for source in sources[1:]:
        file = os.path.join('examples','a1','foot_tracking_gait', source,'weight_1e+03','trajoptresults.pkl')
        data = utils.load(file)
        for key in datakeys:
            fullgait[key] = np.concatenate([fullgait[key], data[key][:, 1:]], axis=1)
        data['time'] += fullgait['time'][-1]
        fullgait['time'] = np.concatenate([fullgait['time'], data['time'][1:]], axis=0)
    # Pop the extra data fields
    poppables = ['solver','success','exit_code','final_cost','duals']
    for poppable in poppables:
        fullgait.pop(poppable)
    return fullgait

def visualize_full_gait(show=True, savename=None):
    if savename and not os.path.isdir(savename):
        os.makedirs(savename)
    # Make the model
    a1 = opttools.make_a1()
    # Get the data and convert to piecewise polynomials
    data = load_full_gait()
    datakeys = getdatakeys()
    for key in datakeys:
        data[key] = PP.FirstOrderHold(data['time'], data[key])
    # Plot the data
    a1.plot_trajectories(xtraj=data['state'], utraj=data['control'], ftraj=data['force'], jltraj=data['jointlimit'], show=show, savename=os.path.join(savename,
    "A1Gait.png"))
    # Plot the foot trajectory
    a1.plot_foot_trajectory(data['state'], show=False, savename=os.path.join(savename,'FootTrajectory.png'))
    # Make a meshcat visualization
    a1.visualize(data['state'])

def concatenate_foot_trajectories(feet):
    foot_ref = []
    for k in range(4):
        foot_ref.append(np.concatenate([foot[k] for foot in feet], axis=1))
    return foot_ref


def main_fullgait_optimization():
    # Get the foot reference trajectory
    savedir = os.path.join('examples','a1','foot_tracking_gait','fullgait')
    a1 = opttools.make_a1()
    feet, _, qtraj = make_a1_step_trajectories(a1)
    # Concatenate the foot trajectories together
    foot_ref = concatenate_foot_trajectories(feet)
    # Create the base trajectory
    base_ref = qtraj.pop()[:6, :]
    for q in qtraj:
        base_ref = np.concatenate([base_ref, q[:6, 1:]], axis=1)
    # Get the warmstart data
    warmstart = load_full_gait()
    duration= warmstart['time'][-1]
    #Change the solver options
    #options = {'Superbasics limit': 2000}
    # Generate the optimization
    optimize_foot_tracking_gait(a1, foot_ref, base_ref, duration, warmstart, savedir)

    
if __name__ == '__main__':
    main_fullgait_optimization()