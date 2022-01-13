"""
Gait trajectory optimization for A1.

Runs trajectory optimization for A1 with a nonlinear foot-tracking cost

Luke Drnach
January 12, 2022
"""
import numpy as np
import os
import matplotlib.pyplot as plt

from pydrake.all import PiecewisePolynomial

from pycito.utilities import save

from make_a1_warmstart import A1GaitGenerator, GaitParameters
import a1trajopttools as opttools

#TODO: Refactor and clean up all the warmstart generating code
#TODO(REFACTOR): factory method to choose between ellipse, quadratic, and quartic
#TODO: Generate successive gait cycles
#TODO: Finish 


def make_ellipse_trajectory(xstart, xgoal, zpeak, samples):
    d = 0.5*(xgoal - xstart)
    x = np.linspace(xstart, xgoal, samples)
    r = 1 - (x - (xstart + d))**2  / (d**2)
    r[r < 0] = 0
    z = zpeak * np.sqrt(r)
    return x, z

def make_quadratic_trajectory(xstart, xgoal, zpeak, samples):
    x, z = make_ellipse_trajectory(xstart, xgoal, 1., samples)
    z = zpeak * (z**2)
    return x, z

def make_quartic_trajectory(xstart, xgoal, zpeak, samples):
    x, z = make_ellipse_trajectory(xstart, xgoal, 1., samples)
    z = zpeak * (z**4)
    return x, z

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
        prog.AddCost(self.eval_cost, vars=qvars, description='FootTrackingCost')
        return prog

    def eval_cost(self, qvars):
        """
        Evaluate the tracking cost
        
        TODO: Rewrite so it works with multiple feet. FOOT_Traj is a list of foot trajectories? / check the type of FOOT_TRAJ
        TODO: FOOT_Q is a list of foot positions, so this won't work
        """
        cost = 0
        a1, context = self._autodiff_or_float(qvars)
        for q, ft in zip(qvars.T, self.foot_traj.T):
            a1.multibody.SetPositions(context, q)
            foot_q = a1.get_foot_positions_in_world(context)
            cost += self.cost_weight * (foot_q - ft).dot(foot_q - ft)
        return cost

    def _autodiff_or_float(self, z):
        if z.dtype == "float":
            return (self.a1_f, self.context_f)
        else:
            return (self.a1_ad, self.context_ad)

def make_a1_foot_trajectory(a1, q0, step_length=0.25, step_height=0.1, swing_phase=0.8, samples=101):
    # Get the starting feet positions
    context = a1.multibody.CreateDefaultContext()
    a1.multibody.SetPositions(context, q0)
    feet = a1.get_foot_position_in_world(context)  #FOOT ORDERING: FL, FR, BL, BR
    # Create a single swing trajectory for FR and BL
    xFR, zFR = make_ellipse_trajectory(feet[1][0], feet[1][0] + step_length, step_height, int(samples * swing_phase))
    xBL, zBL = make_ellipse_trajectory(feet[2][0], feet[2][0] +  step_length, step_height, int(samples * swing_phase))
    stance = (samples - int(swing_phase * samples))//2 
    FR = np.repeat(feet[1], repeats = samples, axis=1)
    BL = np.repeat(feet[2], repeats = samples, axis=1)
    FR[0, stance:stance + xFR.shape[0]] = xFR[:, 0]
    FR[0, stance+xFR.shape[0]:] = xFR[-1, 0]
    FR[2, stance:stance + zFR.shape[0]] = feet[1][2] + zFR[:, 0]
    
    BL[0, stance:stance + xBL.shape[0]] = xBL[:, 0]
    BL[0, stance+xBL.shape[0]:] = xBL[-1, 0]
    BL[2, stance:stance + xBL.shape[0]] = feet[2][2] + zBL[:, 0]

    # Create a static trajectory for FL and BR
    FL = np.repeat(feet[0], repeats=samples, axis=1)
    BR = np.repeat(feet[-1], repeats=samples, axis=1)

    return FL, FR, BL, BR

def make_a1_base_trajectory(q0, feet_traj):
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

def make_configuration_profile(a1, feet, base):
    """
    Run IK To get a configuration profile from the feet and base trajectories
    """
    q = np.zeros((a1.multibody.num_positions(), base.shape[1]))
    q_ = a1.standing_pose()
    feet_pos = [foot[:, 0] for foot in feet]
    q[:, 0], status = a1.foot_pose_ik(base[:, 0], feet_pos, guess=q_)
    if not status:
        print(f"Foot position IK failed at index 0")
    for n in range(1, base.shape[1]):
        feet_pos = [foot[:, n] for foot in feet]
        q[:, n], status = a1.foot_pose_ik(base[:, n], feet_pos, q[:, n-1])
        if not status:
            print(f"Foot position IK failed at index {n}")
    return q

def visualize_foot_trajectory(a1):

    q0 = a1.standing_pose()
    q0, _ = a1.standing_pose_ik(base_pose = q0[:6], guess=q0)
    feet = make_a1_foot_trajectory(a1, q0)
    base = make_a1_base_trajectory(q0, feet)
    q = make_configuration_profile(a1, feet, base)
    t = np.linspace(0, 1, q.shape[1])
    x = np.concatenate([q, np.zeros_like(q)], axis=0)
    xtraj = PiecewisePolynomial.FirstOrderHold(t, x)
    a1.visualize(xtraj)
    # Plot the data
    fig, axs = plt.subplots(3,1)
    labels = ['FL','FR','BL','BR']
    axislabels = ['X','Y','Z']
    for n in range(3):
        for k, foot in enumerate(feet):
            axs[n].plot(t, foot[n,:], linewidth=1.5, label=labels[k])
        axs[n].set_ylabel(axislabels[n])
    axs[-1].set_xlabel('Time (s)')
    axs[0].legend()
    plt.show()


def check_foot_tracking_cost():
    pass


def main():
    a1 = opttools.make_a1()
    visualize_foot_trajectory(a1)

            
    


if __name__ == '__main__':
    main()