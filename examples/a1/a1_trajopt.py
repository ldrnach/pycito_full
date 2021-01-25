"""
Trajectory optimization for A1

Base code for running trajectory optimization with A1.


"""
#Library imports
import numpy as np
import timeit 
from pydrake.solvers.snopt import SnoptSolver

# Project imports
from systems.A1.a1 import A1 
from trajopt.contactimplicit import ContactImplicitDirectTranscription
import utilities as utils

# Create the plant
a1 = A1()
a1.Finalize()
context = a1.multibody.CreateDefaultContext()
# Create the trajectory optimization problem
trajopt = ContactImplicitDirectTranscription(a1, context,
                                num_time_samples=101,
                                minimum_timestep=0.01,
                                maximum_timestep=0.03)
# Create initial and final state constraints
pose = a1.standing_pose()
no_vel = np.zeros((a1.multibody.num_velocities(),))
x0 = np.concatenate((pose, no_vel), axis=0)
xf = x0.copy()
# Set the final pose 1m in front of the initial pose
xf[4] += 1
# Add initial and final state constraints
trajopt.add_state_constraint(knotpoint=0, value=x0)
trajopt.add_state_constraint(knotpoint=100, value=xf)

# Create and add a running cost
numU = a1.multibody.num_actuators()
numX = x0.shape[0]
R = 0.1*np.eye(numU)
u0 = np.zeros((numU,))
Q = np.eye(numX)
# Running costs on state and control effort
trajopt.add_quadratic_running_cost(R, u0, vars=[trajopt.u], name='ControlCost')
trajopt.add_quadratic_running_cost(Q, xf, vars=[trajopt.x], name='StateCost')
# Create the initial guess
u_init = np.zeros(trajopt.u.shape)
l_init = np.zeros(trajopt.l.shape)
x_init = np.linspace(x0, xf, 101)
trajopt.set_initial_guess(x_init, u_init, l_init)

# Create the final program
prog = trajopt.get_program()
# Set SNOPT solver options
prog.SetSolverOption(SnoptSolver().solver_id(), "Iterations Limit", 10000)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Feasibility Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Major Optimality Tolerance", 1e-6)
prog.SetSolverOption(SnoptSolver().solver_id(), "Scale Option", 2)
solver = SnoptSolver()
# Check for bugs in the problem setup
if not utils.CheckProgram(prog):
    quit()

# Solve the problem
print("Solving trajectory optimization")
start = timeit.default_timer()
result = solver.Solve(prog)
stop = timeit.default_timer()
print(f"Elapsed time: {stop-start}")
# Print details of solution
print(f"Optimization successful? {result.is_success()}")
print(f"Solved with {result.get_solver_id().name()}")
print(f"Optimal cost = {result.get_optimal_cost()}")
# Get exit code from SNOPT
print(f"SNOPT Exit Status {result.get_solver_detials().info}: {utils.SNOPT_DECODER[result.get_solver_details().info]}")

# Get the solution trajectories
xtraj, utraj, ltraj, jltraj = trajopt.reconstruct_all_trajectories(result)
a1.plot_trajectories(xtraj, utraj,ltraj, jltraj)
a1.visualize(xtraj)

# Checking for floating base bodies
floating = a1.multibody.GetFloatingBaseBodies()
while len(floating) > 0:
    body = a1.multibody.get_body(floating.pop())
    body.has_quaternion_dofs()
    # Get the indices in the state vector when floating positions and velocities begin
    pos_idx = body.floating_positions_start()
    vel_idx = body.floating_velocities_start()