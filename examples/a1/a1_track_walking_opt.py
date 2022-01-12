"""
Trajectory optimization for A1 to walk forward, tracking a desired joint trajectory

Luke Drnach
December 13, 2021
"""
import numpy as np
import os
from examples.a1 import a1trajopttools as opttools
from examples.a1.make_a1_warmstart import A1GaitGenerator
from pycito.utilities import save
from pydrake.all import PiecewisePolynomial as pp

def get_walking_boundary(a1, distance):
    pose = a1.standing_pose()
    pose, status = a1.standing_pose_ik(base_pose=pose[:6], guess=pose)
    assert status, 'IK failed'
    no_vel = np.zeros((a1.multibody.num_velocities(), ))
    x0 = np.concatenate((pose, no_vel), axis=0)
    xf = x0.copy()
    xf[0] = distance
    return x0, xf

def get_walking_reference_trajectory(N, distance):
    # Make the warmstart
    generator = A1GaitGenerator()
    q, u, fN = generator.make_a1_walking_warmstart(distance, sampling=N)
    # Expand state and forces
    v = np.zeros_like(q)
    x = np.concatenate([q, v], axis=0)
    f = np.zeros((2*generator.num_contacts() + generator.num_friction(), N))
    f[:4, :] = fN
    return x, u, f

def check_walking_reference(N, distance, savedir):
    """
    Check the walking reference trajectory

    """
    a1 = opttools.make_a1()
    warmstartdir = os.path.join(savedir, 'warmstart')
    x, u, f = get_walking_reference_trajectory(N, distance)
    # Save the results
    jl = np.zeros((x.shape[0] - 12, N))
    t = np.linspace(0, 1, N)
    warmstart = {'time': t, 
                'state': x, 
                'control': u,
                'force': f,
                'jointlimit': jl}
    save(os.path.join(warmstartdir, 'warmstart.pkl'), warmstart)
    print(f"Trajectory data saved in {warmstartdir}")
    # Convert the arrays to trajectories
    xtraj = pp.FirstOrderHold(t, x)
    utraj = pp.FirstOrderHold(t, u)
    ftraj = pp.FirstOrderHold(t, f)
    jtraj = pp.FirstOrderHold(t, jl)
    # Make plots of the results
    a1.plot_trajectories(xtraj=xtraj, utraj=utraj, ftraj=ftraj, jltraj=jtraj, show=False, savename=os.path.join(warmstartdir,'warmstart.png'))
    print(f"Trajectory plots saved in {warmstartdir}")
    # Make a visualization of the state
    a1.visualize(xtraj)

def main():
    # Setup the trajectory optimization
    savedir = os.path.join('examples','a1','walking','trackingcost_quarter_51')
    a1 = opttools.make_a1()
    # This setup is for a 1m walk at 0.5m/s with a sampling rate of 50Hz
    N = 51        # Number of knot points
    duration = [1, 1]   # Minimum and maximum trajectory duration 
    distance = 0.25

    trajopt = opttools.make_a1_trajopt(a1, N, duration)
    # Make and add boundary constraints
    x0, xf = get_walking_boundary(a1, distance)
    trajopt = opttools.add_boundary_constraints(trajopt, x0, xf)
    # Make and add the initial guess
    x, u, f = get_walking_reference_trajectory(N, distance)
    trajopt.set_initial_guess(xtraj=x, utraj=u, ltraj=f)
    # Add the running costs
    trajopt = opttools.add_control_cost(trajopt, weight=0.01)
    trajopt = opttools.add_joint_tracking_cost(trajopt, weight=10, qref=x[:a1.multibody.num_positions(), :])
    # Solve the problem
    weights = [1, 10, 100, 1000]
    opttools.progressive_solve(trajopt, weights, savedir)

if __name__ == '__main__':
    #check_walking_reference(N=51, distance = 0.25, savedir = os.path.join('examples','a1','walking','trackingcost_quarter_51'))
    main()
