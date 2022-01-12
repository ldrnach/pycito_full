"""
A1 trajectory optimization tools

Luke Drnach
December 13, 2021
"""

import numpy as np
from pycito.systems.A1.a1 import A1VirtualBase
from pycito.trajopt import contactimplicit as ci
import os
import pycito.utilities as utils

def make_a1():
    a1 = A1VirtualBase()
    a1.terrain.friction = 1.0
    a1.Finalize()
    return a1

def solve_static_equilibrium(a1, qtraj):
    """Generate the associated static equilibrium forces"""
    N = qtraj.shape[1]
    u = np.zeros((a1.multibody.num_actuators(), N))
    fN = np.zeros((a1.num_contacts(), N))
    for n in range(N):
        u[:, n], fN[:, n] = a1.static_controller(qtraj[:, n], verbose=False)
    fT = np.zeros((a1.num_friction() + a1.num_contacts(), N))
    return u, np.concatenate([fN, fT], axis=0)

def make_a1_trajopt(a1, N=101, duration=[1, 1]):
    # Create trajopt
    context = a1.multibody.CreateDefaultContext()
    options = ci.OptimizationOptions()
    options.useNonlinearComplementarityWithCost()
    min_time = duration[0]
    max_time = duration[1]
    trajopt = ci.ContactImplicitDirectTranscription(a1, context,
                                        num_time_samples=N,
                                        minimum_timestep=min_time/(N-1),
                                        maximum_timestep=max_time/(N-1),
                                        options = options)
    # Add equal timestep constraints
    trajopt.add_equal_time_constraints()
    # Set the solver options
    trajopt.setSolverOptions({'Iterations limit': 10000000,
                            'Major iterations limit': 5000,
                            'Major feasibility tolerance': 1e-6,
                            'Major optimality tolerance': 1e-6,
                            'Scale option': 2})
    return trajopt

def make_a1_trajopt_linearcost(a1, N=101, duration=[1, 1]):
    # Create trajopt
    context = a1.multibody.CreateDefaultContext()
    options = ci.OptimizationOptions()
    options.useLinearComplementarityWithCost()
    min_time = duration[0]
    max_time = duration[1]
    trajopt = ci.ContactImplicitDirectTranscription(a1, context,
                                        num_time_samples=N,
                                        minimum_timestep=min_time/(N-1),
                                        maximum_timestep=max_time/(N-1),
                                        options = options)
    # Add equal timestep constraints
    trajopt.add_equal_time_constraints()
    # Set the solver options
    trajopt.setSolverOptions({'Iterations limit': 10000000,
                            'Major iterations limit': 5000,
                            'Major feasibility tolerance': 1e-6,
                            'Major optimality tolerance': 1e-6,
                            'Scale option': 2})
    return trajopt

def plot_and_save(trajopt, results, savedir):
    """
    Plot and save the results of trajectory optimization

    Plots the state, control, reaction force, and joint limit force trajectories for A1

    Arguments:
        trajopt: The trajectory optimization solved
        results: The results structure from the Mathematical Program
        savedir: The location at which to save the figures and program report.
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    text = trajopt.generate_report(results)
    with open(os.path.join(savedir, 'report.txt'), 'w') as file:
        file.write(text)
    utils.printProgramReport(results, trajopt.get_program(), verbose=True)
    file =  os.path.join(savedir, 'trajoptresults.pkl')
    utils.save(file, trajopt.result_to_dict(results))
    x, u, l, jl, s = trajopt.reconstruct_all_trajectories(results)
    trajopt.plant_f.plot_trajectories(x, u, l, jl, show=False, savename=os.path.join(savedir,'A1Opt.png'))

def add_control_cost(trajopt, weight=0.01):
    """
    Add a quadratic control cost to the problem
    """
    R = weight * np.eye(trajopt.u.shape[0])
    b = np.zeros((trajopt.u.shape[0]))
    trajopt.add_quadratic_running_cost(R, b, vars=trajopt.u, name='ControlCost')
    return trajopt

def add_joint_tracking_cost(trajopt, weight, qref):
    """
    Add a quadratic joint (configuration) tracking cost to the problem
    """
    Q = weight * np.eye(qref.shape[0])
    qvars = trajopt.x[:trajopt.plant_f.multibody.num_positions(), :]
    trajopt.add_tracking_cost(Q, qref, vars=qvars, name='JointTracking')
    return trajopt

def add_force_cost(trajopt, weight):
    """
    Add a quadratic cost on the normal forces
    """
    nF = trajopt.numN
    Q = weight * np.eye(nF)
    b = np.zeros((nF, ))
    trajopt.add_quadratic_running_cost(Q, b, vars=trajopt.l[:nF, :], name='ForceCost')
    return trajopt

def add_control_difference_cost(trajopt, weight):
    """
    Add a quadratic cost on the change in control torque
    """
    Q = weight * np.eye(trajopt.u.shape[0])
    trajopt.add_quadratic_differenced_cost(Q, vars=trajopt.u, name='ControlDifference')
    return trajopt

def add_force_difference_cost(trajopt, weight):
    """
    Add a quadratic cost on the change in normal force
    """
    nF = trajopt.numN
    Q = weight * np.eye(nF)
    trajopt.add_quadratic_differenced_cost(Q, vars=trajopt.l[:nF, :], name='ForceDifference')

def add_boundary_constraints(trajopt, x0, xf):
    """
    Add boundary constraints to the trajopt
    """
    trajopt.add_state_constraint(knotpoint=0, value=x0)
    trajopt.add_state_constraint(knotpoint=trajopt.num_time_samples-1, value=xf)
    return trajopt

def progressive_solve(trajopt, weights, savedir):
    """
    Solve a sequence of trajectory optimization problems, each with a different complementarity cost weight

    Arguments:
        trajopt: The trajectory optimization problem
        weights: an iterable containing the complementarity cost weights (typically monotonically increasing)
        savedir: the location at which the results of trajectory optimization will be saved
    """
    trajopt.enable_cost_display('figure')
    for weight in weights:
        print(f"Running optimization with weight {weight:.0e}")
        savedir_this = os.path.join(savedir, f"weight_{weight:.0e}")
        trajopt.complementarity_cost_weight = weight
        results = trajopt.solve()
        plot_and_save(trajopt, results, savedir_this)
        trajopt.initialize_from_previous(results)
        trajopt.printer.save_and_clear(savename=os.path.join(savedir_this, 'CostsAndConstraints.png'))