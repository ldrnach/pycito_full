"""
Trajectory optimization for A1 to lift, tracking a desired trajectory

Luke Drnach
December 10, 2021
"""
import numpy as np
from pydrake.all import PiecewisePolynomial
from pycito.trajopt import contactimplicit as ci
import os
from examples.a1 import a1trajopttools as opttools

def get_lifting_boundary(a1):
    pose = a1.standing_pose()
    pose2 = pose.copy()
    pose[2] /= 2
    pose_ik, status = a1.standing_pose_ik(pose[:6], guess=pose.copy())
    assert status, "IK failed for initial pose"
    pose2_ik, status = a1.standing_pose_ik(pose2[:6], guess=pose2.copy())
    assert status, "IK failed for final pose"
    return pose_ik, pose2_ik

def lifting_reference_trajectory(a1, base0, baseF, N):
    """Generate a smooth lifting trajectory for a1"""    
    # Get the base trajectory
    spline = PiecewisePolynomial.CubicHermite([0, 1], np.column_stack([base0, baseF]), np.zeros((6,2)))
    t = np.linspace(0, 1, N)
    basetraj = spline.vector_values(t)
    # Solve for the configuration trajectory
    q0 = a1.standing_pose()
    q = np.zeros((a1.multibody.num_positions(), N))
    for n in range(N):
        q[:, n], status = a1.standing_pose_ik(base_pose = basetraj[:, n], guess = q0)
        assert status, f"IK failed at point {n}"
        q0 = q[:, n]
    return q
    
def test_lifting_reference():
    N = 101
    a1 = opttools.make_a1()
    pose1, pose2 = get_lifting_boundary(a1)
    q = lifting_reference_trajectory(a1, pose1[:6], pose2[:6], N)
    u, f = opttools.solve_static_equilibrium(a1, q)
    v = np.zeros((a1.multibody.num_velocities(), N))
    x = np.concatenate((q, v), axis=0)
    t = np.linspace(0, 1, N)
    xtraj = PiecewisePolynomial.FirstOrderHold(t, x)
    utraj = PiecewisePolynomial.ZeroOrderHold(t, u)
    ftraj = PiecewisePolynomial.ZeroOrderHold(t, f)
    a1.visualize(xtraj)
    a1.plot_trajectories(xtraj, utraj, ftraj, show=True)
    
def make_reference_trajectories(a1, N):
    q0, qf = get_lifting_boundary(a1)
    q = lifting_reference_trajectory(a1, q0[:6], qf[:6], N)
    u, f = opttools.solve_static_equilibrium(a1, q)
    v = np.zeros((a1.multibody.num_velocities(), N))
    x = np.concatenate((q, v), axis=0)
    return x, u, f

def main():
    savedir = os.path.join('examples','a1','lift_test','trackingcost')
    a1 = opttools.make_a1()
    N = 21
    trajopt = opttools.make_trajopt(a1, N, duration=[1, 1])
    # Make and add boundary conditions
    x0 = np.zeros((trajopt.x.shape[0], ))
    xf = np.zeros((trajopt.x.shape[0], ))
    x0[:a1.multibody.num_positions()], xf[:a1.multibody.num_positions()] = get_lifting_boundary(a1)
    opttools.add_boundary_constraints(trajopt, x0, xf)
    # Make the reference trajectory and set the initial guess
    x, u, f = make_reference_trajectories(a1, N)
    trajopt.set_initial_guess(xtraj=x, utraj=u, ltraj=f)
    # Add the costs
    opttools.add_control_cost(trajopt, weight=0.01)
    opttools.add_joint_tracking_cost(trajopt, weight=10, qref = x[:a1.multibody.num_positions(), :])
    # Solve the problem
    weights = [1, 10, 100, 1000]
    opttools.progressive_solve(trajopt, weights, savedir)

if __name__ == '__main__':
    main()