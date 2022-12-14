"""
Trajectory Optimization for A1: Goal is to solve for a static pose
Luke Drnach
February 26, 2021
"""
import numpy as np
import timeit
from pycito.trajopt.contactimplicit import ContactImplicitDirectTranscription, OptimizationOptions 
from pycito.systems.A1.a1 import A1VirtualBase
import pycito.utilities as utils
from pydrake.all import PiecewisePolynomial
from pydrake.solvers.snopt import SnoptSolver

# Create the plant for A1 and the associated trajectory optimization
a1 = A1VirtualBase()
a1.terrain.friction = 1.0
a1.Finalize()
context = a1.multibody.CreateDefaultContext()
options = OptimizationOptions()
options.useNonlinearComplementarityWithCost()
trajopt = ContactImplicitDirectTranscription(a1, context,
                                    num_time_samples=21,
                                    minimum_timestep=0.1,
                                    maximum_timestep=0.1,
                                    options=options)
trajopt.complementarity_cost_weight = 1.
# Create and set boundary conditions
pose = a1.standing_pose()
pose2 = pose.copy()
# Append zeros to make a full state
no_vel = np.zeros((a1.multibody.num_velocities(), ))
x0 = np.concatenate((pose, no_vel),axis=0)
xf = np.concatenate((pose2, no_vel), axis=0)
# linear interpolate so we can visualize
x_init = np.linspace(x0, xf, trajopt.num_time_samples).transpose()
# Create a PiecewisePolynomial for visualization
t_init = np.linspace(0, 1, trajopt.num_time_samples)
x_traj = PiecewisePolynomial.FirstOrderHold(t_init, x_init)
# a1.visualize(x_traj)
# quit()
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
trajopt.set_initial_guess(xtraj=x_init, utraj=u_init, ltraj=l_init, jltraj=jl_init)
# Add a running cost on the controls - without the cost, A1 might not remain static
R = 0.01*np.eye(uref.shape[0])
trajopt.add_quadratic_running_cost(R, uref, vars=trajopt.u, name="ControlCost")
# Check the program
if not utils.CheckProgram(trajopt.prog):
    quit()
# Set SNOPT solver options
trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 1000000)
trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
trajopt.prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
trajopt.enable_cost_display('figure')
#trajopt.enable_iteration_visualizer()
solver = SnoptSolver()
print("Solving trajectory optimization")
start = timeit.default_timer()
result = solver.Solve(trajopt.get_program())
stop = timeit.default_timer()
print(f"Elapsed time: {stop-start}")
# Print details of solution
utils.printProgramReport(result, trajopt.get_program())
file = 'data/a1/a1_virtual_static_regularized_opt_06152021.pkl'
utils.save(file, trajopt.result_to_dict(result))
x, u, l, jl, s = trajopt.reconstruct_all_trajectories(result)
a1.plot_trajectories(x, u, l, jl)
a1.visualize(x)