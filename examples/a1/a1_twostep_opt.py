"""
TwoStep trajectory optimization for A1 with nonlinear foot-tracking

Luke Drnach
February 8, 2022
"""
import numpy as np
import os
import matplotlib.pyplot as plt

from pydrake.all import PiecewisePolynomial as PP

import pycito.utilities as utils
import a1trajopttools as opttools
import a1_foot_tracking_opt as trackingopt

def concatenate_foot_trajectories(feet):
    foot_ref = feet.pop(0)
    for foot in feet:
        for k in range(4):
            foot_ref[k] = np.concatenate([foot_ref[k], foot[k][:, 1:]], axis=1)
    return foot_ref

def plot_foot_trajectories(a1, feet):
    labels = a1.foot_frame_names
    _, axs = plt.subplots(3,1)
    for foot, label in zip(feet, labels):
        for n in range(3):
            axs[n].plot(foot[n,:], linewidth=1.5, label=label)
    axs[0].set_ylabel('X')
    axs[0].set_ylabel('Y')
    axs[2].set_ylabel('Z')
    axs[2].set_xlabel('Time index')
    axs[0].legend()
    plt.show()

def periodicity_parameters(dim):
    return np.concatenate([np.eye(dim), -np.eye(dim)], axis=1), np.zeros((dim, ))

def twostep_main():
    savedir = os.path.join('examples','a1','foot_tracking_gait','twostepopt')
    # Get the foot reference trajectory
    a1 = opttools.make_a1()
    feet, _, qtraj = trackingopt.make_a1_step_trajectories(a1)
    foot_ref = concatenate_foot_trajectories(feet[1:3])
    qtraj = qtraj[1:3]
    base_ref = qtraj.pop(0)[:6, :]
    for q in qtraj:
        base_ref = np.concatenate([base_ref, q[:6, 1:]], axis=1)
    # Load the warmstart
    warmstart = utils.load(utils.FindResource(os.path.join('examples','a1','foot_tracking_gait','twostep_plots','combinedresults.pkl')))
    duration = warmstart['time'][-1]
    # Setup the trajopt
    #options = {'Iterations limit': 1}
    trajopt = trackingopt.setup_foot_tracking_gait(a1, foot_ref, base_ref, duration, warmstart)
    # Add periodicity constraints
    Au, bu = periodicity_parameters(trajopt.u.shape[0])
    Al, bl = periodicity_parameters(trajopt.l.shape[0])
    Aj, bj = periodicity_parameters(trajopt.jl.shape[0])
    trajopt.prog.AddLinearEqualityConstraint(Au, bu, np.concatenate([trajopt.u[:, 0], trajopt.u[:, -1]], axis=0)).evaluator().set_description('control_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Al, bl, np.concatenate([trajopt.l[:, 0], trajopt.l[:, -1]], axis=0)).evaluator().set_description('force_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Aj, bj, np.concatenate([trajopt.jl[:, 0], trajopt.jl[:, -1]], axis=0)).evaluator().set_description('jointlimit_periodicity')
    # Solve the problem
    weights = [1, 1e1, 1e2, 1e3]
    opttools.progressive_solve(trajopt, weights, savedir)

def twostep_sanity_check():
    # Get the foot reference trajectory
    a1 = opttools.make_a1()
    feet, _, qtraj = trackingopt.make_a1_step_trajectories(a1)
    foot_ref = concatenate_foot_trajectories(feet[1:3])
    qtraj = qtraj[1:3]
    base_ref = qtraj.pop(0)[:6, :]
    for q in qtraj:
        base_ref = np.concatenate([base_ref, q[:6, 1:]], axis=1)
    # Load the warmstart
    warmstart = utils.load(utils.FindResource(os.path.join('examples','a1','foot_tracking_gait','twostepopt','restart','weight_1e+03','trajoptresults.pkl')))
    duration = warmstart['time'][-1]
    # Setup the trajopt
    options = {'Iterations limit': 1}
    trajopt = trackingopt.setup_foot_tracking_gait(a1, foot_ref, base_ref, duration, warmstart)
    # Add periodicity constraints
    Au, bu = periodicity_parameters(trajopt.u.shape[0])
    Al, bl = periodicity_parameters(trajopt.l.shape[0])
    Aj, bj = periodicity_parameters(trajopt.jl.shape[0])
    trajopt.prog.AddLinearEqualityConstraint(Au, bu, np.concatenate([trajopt.u[:, 0], trajopt.u[:, -1]], axis=0)).evaluator().set_description('control_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Al, bl, np.concatenate([trajopt.l[:, 0], trajopt.l[:, -1]], axis=0)).evaluator().set_description('force_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Aj, bj, np.concatenate([trajopt.jl[:, 0], trajopt.jl[:, -1]], axis=0)).evaluator().set_description('jointlimit_periodicity')
    trajopt.complementarity_cost_weight = 1e3
    # First, check all the constraint values
    printer = utils.MathProgIterationPrinter(trajopt.prog)
    vals = trajopt.prog.GetInitialGuess(trajopt.prog.decision_variables())
    costs = printer.calc_costs(vals)
    cstrs = printer.calc_constraints(vals)
    print('Cost Function Values:')
    for key, value in costs.items():
        print(f"\t {key}: {value:.4E}")
    print("Constriant violations:")
    for key, value in cstrs.items():
        print(f"\t {key}: {value:.4E}")

def twostep_sanity_restart():
    # Get the foot reference trajectory
    a1 = opttools.make_a1()
    feet, _, qtraj = trackingopt.make_a1_step_trajectories(a1)
    foot_ref = concatenate_foot_trajectories(feet[1:3])
    qtraj = qtraj[1:3]
    base_ref = qtraj.pop(0)[:6, :]
    for q in qtraj:
        base_ref = np.concatenate([base_ref, q[:6, 1:]], axis=1)
    # Load the warmstart
    warmstart = utils.load(utils.FindResource(os.path.join('examples','a1','foot_tracking_gait','twostepopt','weight_1e+03','trajoptresults.pkl')))
    duration = warmstart['time'][-1]
    # Setup the trajopt
    trajopt = trackingopt.setup_foot_tracking_gait(a1, foot_ref, base_ref, duration, warmstart)
    # Add periodicity constraints
    Au, bu = periodicity_parameters(trajopt.u.shape[0])
    Al, bl = periodicity_parameters(trajopt.l.shape[0])
    Aj, bj = periodicity_parameters(trajopt.jl.shape[0])
    trajopt.prog.AddLinearEqualityConstraint(Au, bu, np.concatenate([trajopt.u[:, 0], trajopt.u[:, -1]], axis=0)).evaluator().set_description('control_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Al, bl, np.concatenate([trajopt.l[:, 0], trajopt.l[:, -1]], axis=0)).evaluator().set_description('force_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Aj, bj, np.concatenate([trajopt.jl[:, 0], trajopt.jl[:, -1]], axis=0)).evaluator().set_description('jointlimit_periodicity')
    # Solve just the last optimization
    weights = [1e3]
    savedir = os.path.join('examples','a1','foot_tracking_gait','twostepopt','restart')
    opttools.progressive_solve(trajopt, weights, savedir)


if __name__ == "__main__":
    twostep_sanity_check()