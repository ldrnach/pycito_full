"""
Trajectory optimization for A1

Base code for running trajectory optimization with A1.

Luke Drnach
February 2021
"""
#Library imports
import numpy as np
import timeit 
from pydrake.solvers.snopt import SnoptSolver
# Project imports
from pycito.systems.A1.a1 import A1 
from pycito.trajopt.contactimplicit import ContactImplicitDirectTranscription
import pycito.utilities as utils

def run_a1_trajopt():
    """ Run trajectory optimization for A1 quadruped"""
    # Create A1 plant and trajectory optimization
    trajopt, a1 = setup_a1_trajopt()
    # Create and set boundary states for A1
    x0, xf = get_a1_boundary_conditions(a1)
    set_a1_boundary_conditions(trajopt, x0, xf)
    # Create and set the initial guess
    guess = create_linear_guess(trajopt, x0, xf)
    set_initial_guess(trajopt, guess)
    # Add running costs
    # u0, _ = a1.static_controller(qref=x0[0:a1.multibody.num_positions()])
    # add_control_cost(trajopt, uref=u0)
    # add_state_cost(trajopt, xref=xf)
    # add_base_tracking_cost(trajopt, guess['state'])
    # Set SNOPT Tolerances and options
    set_default_snopt_options(trajopt)
    # Check the program
    if not utils.CheckProgram(trajopt.prog):
        quit()
    # Solve the problem
    trajopt.enable_cost_display()
    slacks = [10., 1., 0.1, 0.01, 0.001, 0.0001, 0.]
    for slack in slacks:
        # Set the slack variable
        trajopt.set_slack(slack)
        # Solve the optimization
        print(f"NCC Slack = {slack}")
        result = solve_a1_trajopt(trajopt)
        # Get the solution as a dictionary
        soln = trajopt.result_to_dict(result)
        # Initialize to new trajectory
        set_initial_guess(trajopt, soln)
        # Save the results
        save_a1_results(soln, name=f'a1_trajopt_slack_{slack:.0e}.pkl')

def setup_a1_trajopt():
    """Create an A1 Plant and associated trajectory optimization"""
    a1 = A1()
    a1.Finalize()
    context = a1.multibody.CreateDefaultContext()
    trajopt = ContactImplicitDirectTranscription(a1, context,
                                    num_time_samples=101,
                                    minimum_timestep=0.01,
                                    maximum_timestep=0.03)
    return trajopt, a1

def get_a1_boundary_conditions(a1):
    """ Boundary states for A1 trajectory optimization"""
    pose = a1.standing_pose()
    no_vel = np.zeros((a1.multibody.num_velocities(), ))
    x0 = np.concatenate((pose, no_vel), axis=0)
    xf = x0.copy()
    # Set the final pose 1m in front of the starting pose
    xf[4] += 1.
    return x0, xf

def set_a1_boundary_conditions(trajopt, x0, xf):
    """ Set the boundary conditions in the trajopt"""
    trajopt.add_state_constraint(knotpoint=0, value=x0)
    trajopt.add_state_constraint(knotpoint=trajopt.num_time_samples-1, value=xf)

def add_control_cost(trajopt, uref=None):
    """ Add a quadratic running cost on the control effort"""
    # Create and add a running cost
    numU = trajopt.u.shape[0]
    R = 0.1*np.eye(numU)
    if uref is None:
        uref = np.zeros((numU,))
    # Running costs on state and control effort
    trajopt.add_quadratic_running_cost(R, uref, vars=[trajopt.u], name='ControlCost')

def add_state_cost(trajopt, xref=None):
    """Add a quadratic running cost on the state """
    #TODO: Implement only for joint angles and velocities
    # Create and add a running cost
    numX = trajopt.plant_f.multibody.num_positions() + trajopt.plant_f.multibody.num_velocities()
    pos_weights = [0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    vel_weights = [0., 0., 0., 0.1, 0.1, 0.1, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    Q = np.diag(np.concatenate((pos_weights, vel_weights), axis=0))
    if xref is None:
        xref = np.zeros((numX, ))
    # Running costs on state and control effort
    trajopt.add_quadratic_running_cost(Q, xref, vars=[trajopt.x], name='StateCost')

def add_base_tracking_cost(trajopt, xref):
    """Add a running cost for base tracking """
    base_ref = xref[4:7,:]
    Q = np.eye(3)
    trajopt.add_tracking_cost(Q=Q, traj=base_ref, vars=[trajopt.x[4:7,:]], name="BaseTracking")

def create_linear_guess(trajopt, x0, xf):
    """Create a linear guess for the initial condition"""
    return {
        'control': np.zeros(trajopt.u.shape),
        'force': np.zeros(trajopt.l.shape),
        'state': np.linspace(x0, xf, trajopt.num_time_samples).transpose(),
        'jointlimit': np.zeros(trajopt.jl.shape)
    }

def set_initial_guess(trajopt, guess):
    """Set the initial trajectory guess from a dictionary of values """
    trajopt.set_initial_guess(xtraj=guess['state'], 
                            utraj=guess['control'],
                            ltraj=guess['force'],
                            jltraj=guess['jointlimit'])

def set_default_snopt_options(trajopt):
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 10000)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Elastic Weight", 10e5)
    trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Superbasics Limit",10000)

def solve_a1_trajopt(trajopt):
    solver = SnoptSolver()
    prog = trajopt.get_program()
    # Solve the problem
    print("Solving trajectory optimization")
    start = timeit.default_timer()
    result = solver.Solve(prog)
    stop = timeit.default_timer()
    print(f"Elapsed time: {stop-start}")
    # Print details of solution
    utils.printProgramReport(result, prog)
    return result

def save_a1_results(soln, name='a1_trajopt.pkl'):
    file = "data/a1/" + name
    utils.save(file, soln)

if __name__ == "__main__":
    run_a1_trajopt()

# # Create the plant
# a1 = A1()
# a1.Finalize()
# context = a1.multibody.CreateDefaultContext()
# # Create the trajectory optimization problem
# trajopt = ContactImplicitDirectTranscription(a1, context,
#                                 num_time_samples=101,
#                                 minimum_timestep=0.01,
#                                 maximum_timestep=0.03)
# # Create initial and final state constraints
# pose = a1.standing_pose()
# no_vel = np.zeros((a1.multibody.num_velocities(),))
# x0 = np.concatenate((pose, no_vel), axis=0)
# xf = x0.copy()
# # Set the final pose 1m in front of the initial pose
# xf[4] += 1.
# # Add initial and final state constraints
# trajopt.add_state_constraint(knotpoint=0, value=x0)
# trajopt.add_state_constraint(knotpoint=100, value=xf)

# # Create and add a running cost
# numU = a1.multibody.num_actuators()
# numX = x0.shape[0]
# R = 0.1*np.eye(numU)
# u0 = np.zeros((numU,))
# Q = np.eye(numX)
# # Running costs on state and control effort
# trajopt.add_quadratic_running_cost(R, u0, vars=[trajopt.u], name='ControlCost')
# trajopt.add_quadratic_running_cost(Q, xf, vars=[trajopt.x], name='StateCost')
# # Create the initial guess
# u_init = np.zeros(trajopt.u.shape)
# l_init = np.zeros(trajopt.l.shape)
# x_init = np.linspace(x0, xf, 101).transpose()
# trajopt.set_initial_guess(x_init, u_init, l_init)

# # Create the final program
# prog = trajopt.get_program()
# # Set SNOPT solver options
# prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 10000)
# prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
# prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
# prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
# solver = SnoptSolver()
# # Check for bugs in the problem setup

# # Solve the problem
# print("Solving trajectory optimization")
# start = timeit.default_timer()
# result = solver.Solve(prog)
# stop = timeit.default_timer()
# print(f"Elapsed time: {stop-start}")
# # Print details of solution
# utils.printProgramReport(result, prog)

# # Get the solution trajectories
# xtraj, utraj, ltraj, jltraj = trajopt.reconstruct_all_trajectories(result)
# a1.plot_trajectories(xtraj, utraj, ltraj, jltraj)
# a1.visualize(xtraj)
