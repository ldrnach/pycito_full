"""
Trajectory Optimization for A1: Goal is to complete a short lift
Luke Drnach
June 22, 2021
"""
#TODO: Create a general optimizer class that also handles keeping track of run settings and reporting
import numpy as np
import timeit, os
from trajopt.contactimplicit import ContactImplicitDirectTranscription, OptimizationOptions 
from systems.A1.a1 import A1VirtualBase
import utilities as utils
from pydrake.all import PiecewisePolynomial
from pydrake.solvers.snopt import SnoptSolver
import matplotlib.pyplot as plt

#TODO: Initialize from the standing optimization solution

def create_guess_from_previous(data, num_samples):
    xtraj = create_guess(data['time'], data['state'], num_samples)
    utraj = create_guess(data['time'], data['control'], num_samples)
    ltraj = create_guess(data['time'], data['force'], num_samples)
    jltraj = create_guess(data['time'], data['jointlimit'], num_samples)
    return xtraj, utraj, ltraj, jltraj

def create_guess(time, data, num_samples):
    traj = PiecewisePolynomial.FirstOrderHold(time, data)
    new_time = np.linspace(0, traj.end_time(), num_samples)
    return traj.vector_values(new_time) 

# Create the plant for A1 and the associated trajectory optimization
a1 = A1VirtualBase()
a1.terrain.friction = 1.0
a1.Finalize()
context = a1.multibody.CreateDefaultContext()
options = OptimizationOptions()
options.useNonlinearComplementarityWithCost()
# Trajopt parameters
N = 21
max_time = 2.
min_time = 1.

trajopt = ContactImplicitDirectTranscription(a1, context,
                                    num_time_samples=N,
                                    minimum_timestep=min_time/(N-1),
                                    maximum_timestep=max_time/(N-1),
                                    options=options)
#trajopt.enforceNormalDissipation()
trajopt.complementarity_cost_weight = 100.
trajopt.add_equal_time_constraints()
# Create and set boundary conditions
pose = a1.standing_pose()
pose2 = pose.copy()
# Set the final height
pose2[2] = pose2[2]/2
# Solve for a feasible second pose
pose2_ik, status = a1.standing_pose_ik(base_pose = pose2[0:6], guess = pose2.copy())
# Append zeros to make a full state
no_vel = np.zeros((a1.multibody.num_velocities(), ))
x0 = np.concatenate((pose2_ik, no_vel),axis=0)
xf = np.concatenate((pose, no_vel), axis=0)

# linear interpolate so we can visualize
x_init = np.linspace(x0, xf, trajopt.num_time_samples).transpose()
# Create a PiecewisePolynomial for visualization
t_init = np.linspace(0, 1, trajopt.num_time_samples)
x_traj = PiecewisePolynomial.FirstOrderHold(t_init, x_init)
# Check the trajectory
# Set the final pose 1m in front of the starting pose
trajopt.add_state_constraint(knotpoint=0, value=x0)
trajopt.add_state_constraint(knotpoint=trajopt.num_time_samples-1, value=xf)
# Create a reference standing controller value
uref, _ = a1.static_controller(qref=x0[0:a1.multibody.num_positions()])
# Create and set an initial guess
u_init = np.linspace(uref, uref, trajopt.num_time_samples).transpose()
x_init = np.linspace(x0, xf, trajopt.num_time_samples).transpose()
l_init = np.zeros(trajopt.l.shape)
jl_init = np.zeros(trajopt.jl.shape)
# Intialize from the static case
# data = utils.load("data/a1/a1_virtual_static_regularized_opt_06152021.pkl")
# _, u_init, l_init, jl_init = create_guess_from_previous(data, N)
# x_init = np.linspace(x0, xf,trajopt.num_time_samples).transpose()

trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init, jltraj=jl_init)
# Add a running cost on the controls - without the cost, A1 might not remain static
R = 0.01*np.eye(uref.shape[0])
trajopt.add_quadratic_running_cost(R, uref, vars=trajopt.u, name="ControlCost")
# add a small cost on the distance to the goal to regularize the motion
Q = 10*np.eye((x0.shape[0]))
trajopt.add_quadratic_running_cost(Q, xf, vars=[trajopt.x], name="StateCost")
# Check the program
if not utils.CheckProgram(trajopt.prog):
    quit()
# Set SNOPT solver options
solver = SnoptSolver()
trajopt.prog.SetSolverOption(solver.solver_id(), "Iterations Limit", 1000000)
trajopt.prog.SetSolverOption(solver.solver_id(), "Major iterations limit", 0)
trajopt.prog.SetSolverOption(solver.solver_id(), "Major Feasibility Tolerance", 1e-6)
#trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Minor Feasibility Tolerance", 1e-6)
trajopt.prog.SetSolverOption(solver.solver_id(), "Major Optimality Tolerance", 1e-6)
trajopt.prog.SetSolverOption(solver.solver_id(), "Scale Option", 2)
outfile = os.path.abspath("/workspaces/pyCITO/examples/a1/snopt.txt")
trajopt.prog.SetSolverOption(solver.solver_id(), "Print file", outfile)
trajopt.enable_cost_display('figure')
#trajopt.enable_iteration_visualizer()
print("Solving trajectory optimization")
start = timeit.default_timer()
result = solver.Solve(trajopt.get_program())
stop = timeit.default_timer()
print(f"Elapsed time: {stop-start}")
# Print details of solution
#utils.printProgramReport(result, trajopt.get_program(), verbose=True, filename='examples/a1/figures/a1_lift_06292021/report.txt')
#file = 'data/a1/a1_lift_06292021.pkl'
# utils.save(file, trajopt.result_to_dict(result))
# x, u, l, jl, s = trajopt.reconstruct_all_trajectories(result)
# a1.plot_trajectories(x, u, l, jl)
# a1.visualize(x)