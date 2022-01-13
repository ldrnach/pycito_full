"""
Gait trajectory optimization for A1.

Runs three trajectory optimizations:
    1. The first step
    2. A full gait cycle
    3. The final step

Luke Drnach
January 12, 2022
"""
import numpy as np
import os

from pydrake.all import PiecewisePolynomial

from pycito.utilities import save

from make_a1_warmstart import A1GaitGenerator, GaitParameters
import a1trajopttools as opttools

def make_and_solve_trajopt(a1, xref, uref, fref, duration, savedir):
    """Generically solves the trajectory following optimization"""
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    N = xref.shape[1]
    # Setup the trajectory optimization
    trajopt = opttools.make_a1_trajopt_linearcost(a1, N, duration)
    # Add boundary constraints
    trajopt = opttools.add_boundary_constraints(trajopt, xref[:, 0], xref[:, -1])
    # Set the initial guess
    trajopt.set_initial_guess(xtraj=xref, utraj=uref, ltraj=fref)
    # Require joint tracking                               
    trajopt = opttools.add_joint_tracking_cost(trajopt, weight=1e4, qref=xref[:a1.multibody.num_positions(), :])
    # Set a small cost on control
    trajopt = opttools.add_control_cost(trajopt, weight=1e-2)
    # Add small cost on force
    trajopt = opttools.add_force_cost(trajopt, weight=1e-2)
    # Add small cost on control difference
    trajopt = opttools.add_control_difference_cost(trajopt, weight=1e-2)
    # Add small cost on force difference
    trajopt = opttools.add_force_difference_cost(trajopt, weight=1e-2)
    # Solve the problem using different complementarity cost weights
    weights = [1, 1e1, 1e2, 1e3]
    opttools.progressive_solve(trajopt, weights, savedir)

def visualize_and_save_warmstart(a1, x, u, f, savedir):
    savedir = os.path.join(savedir, 'warmstart')
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    #Visualize the trajectory
    t = np.linspace(0, 1, x.shape[1])
    xtraj = PiecewisePolynomial.FirstOrderHold(t, x)
    a1.visualize(xtraj)
    # Plot the trajectories and save them 
    utraj = PiecewisePolynomial.ZeroOrderHold(t, u)
    ftraj = PiecewisePolynomial.ZeroOrderHold(t, f)
    a1.plot_trajectories(xtraj=xtraj, utraj=utraj, ftraj=ftraj, show=False, savename = os.path.join(savedir, 'warmstart.png'))
    # Save the warmstart data
    warmstart = {'time': t,
                 'state': x,
                 'control': u,
                 'force': f,
                'jointlimit': np.zeros((x.shape[0] - 12, x.shape[1]))}
    filename = os.path.join(savedir, 'warmstart.pkl')
    save(filename, warmstart)
    print(f'Trajectory data saved in {filename}')

def first_step_optimization(savedir, visstart=False):
    """
    Optimize for the first step in the gait cycle
    """
    savedir = os.path.join(savedir, 'firststep')
    # Get the warmstart
    a1 = opttools.make_a1()
    generator = A1GaitGenerator()
    gait = GaitParameters(step_length=0.4)
    q, u, fN = generator.make_first_step_warmstart(gait=gait, sampling=31)
    f = np.zeros((2*a1.num_contacts() + a1.num_friction(), fN.shape[1]))
    f[:fN.shape[0], :] = fN
    v = np.zeros(q.shape)
    x = np.concatenate([q, v], axis=0)
    if visstart:
        # Visualize the warmstart
        visualize_and_save_warmstart(a1, x, u, f, savedir)
    else:
        # Solve the optimization
        make_and_solve_trajopt(a1, x, u, f, [0.3, 0.3], savedir)
    

def full_cycle_optimization(savedir, visstart=False):
    """
    Optimize a full gait cycle
    """    
    savedir = os.path.join(savedir,'fullcycle')
    # Get the warmstart
    a1 = opttools.make_a1()
    generator = A1GaitGenerator()
    gait = GaitParameters(step_length=0.4)
    q, u, fN = generator.make_full_cycle_warmstart(gait=gait, sampling=61)
    f = np.zeros((2*a1.num_contacts() + a1.num_friction(), fN.shape[1]))
    f[:fN.shape[0], :] = fN
    v = np.zeros(q.shape)
    x = np.concatenate([q, v], axis=0)
    if visstart:
        # Visualize the warmstart
        visualize_and_save_warmstart(a1, x, u, f, savedir)
    else:
        # Solve the optimization
        make_and_solve_trajopt(a1, x, u, f, [0.6, 0.6], savedir)

def last_step_optimization(savedir, visstart=False):
    """
    Optimize the last step in the gait cycle
    """
    savedir = os.path.join(savedir, 'laststep')
    # Get the warmstart
    a1 = opttools.make_a1()
    generator = A1GaitGenerator()
    gait = GaitParameters(step_length=0.4)
    q, u, fN = generator.make_last_step_warmstart(gait=gait, sampling=31)
    f = np.zeros((2*a1.num_contacts() + a1.num_friction(), fN.shape[1]))
    f[:fN.shape[0], :] = fN
    v = np.zeros(q.shape)
    x = np.concatenate([q, v], axis=0)
    if visstart:
        # Visualize the warmstart
        visualize_and_save_warmstart(a1, x, u, f, savedir)
    else:
        # Solve the optimization
        make_and_solve_trajopt(a1, x, u, f, [0.3, 0.3], savedir)


if __name__ == '__main__':
    savedir = os.path.join(os.getcwd(), 'examples', 'a1', 'gaitoptimization','tracking_cost_1e4')
    first_step_optimization(savedir, visstart=False)
    full_cycle_optimization(savedir, visstart=False)
    last_step_optimization(savedir, visstart=False)