"""
Contact Implicit Trajectory Optimization for a sliding block. 
This script is used mainly to test the implementation of ContactImplicitOrthogonalCollocation in contactimplicit.py

The goal is to move a 1kg block 5m in 1s. The timesteps are fixed, and the objective is to minimize the control cost and the state deviation from the final position. Boundary constraints are added to the problem to ensure the block starts and stops at rest and at the desired positions. In this example, the timesteps are fixed and equal. 

Luke Drnach
November 5, 2021
"""

# Imports
import os
import numpy as np
import trajopt.contactimplicit as ci
from systems.block.block import Block
import utilities as utils

# Create the block plant
plant = Block()
plant.Finalize()
# Get the default context
context = plant.multibody.CreateDefaultContext()
options = ci.OrthogonalOptimizationOptions()
options.useComlementarityWithConstantSlack()
# Create a Contact Implicit OrthogonalCollocation
N = 26
max_time = 1
min_time = 1

trajopt = ci.ContactImplicitOrthogonalCollocation(plant, context, 
                                                num_time_samples = N, 
                                                minimum_timestep=min_time/(N-1), 
                                                maximum_timestep=max_time/(N-1),
                                                state_order=3,
                                                options=options)
#Additional constraints
trajopt.add_equal_time_constraints()
trajopt.add_zero_acceleration_boundary_condition()
# Boundary conditions
x0 = np.array([0, 0.5, 0., 0.])
xf = np.array([5., 0.5, 0., 0.])
Ntotal = trajopt.total_knots
trajopt.add_state_constraint(knotpoint=0, value=x0)
trajopt.add_state_constraint(knotpoint=Ntotal-1, value=xf)
# Add cost functions
R = 10 * np.eye(trajopt.u.shape[0])
b = np.zeros((trajopt.u.shape[0], ))
trajopt.add_quadratic_control_cost(R, b)
Q = np.eye(trajopt.x.shape[0])
trajopt.add_quadratic_running_cost(Q, xf, vars=[trajopt.x], name='StateCost')
# Set the initial guess
uinit = np.zeros(trajopt.u.shape)
xinit = np.zeros(trajopt.x.shape)
for n in range(0, xinit.shape[0]):
    xinit[n,:] = np.linspace(start=x0[n], stop=xf[n], num=Ntotal)
linit = np.zeros(trajopt.l.shape)
linit[0,:] = 10
sinit = np.zeros(trajopt.var_slack.shape)
ainit = np.zeros(trajopt.accel.shape)
hinit = np.ones(trajopt.h.shape) * max_time/(N-1)
# Set the initial guess
trajopt.set_initial_guess(xtraj=xinit, utraj=uinit, ltraj=linit, straj=sinit)
trajopt.prog.SetInitialGuess(trajopt.accel, ainit)
trajopt.prog.SetInitialGuess(trajopt.h, hinit)
# Check the program
print(f"Checking program")
if not utils.CheckProgram(trajopt.prog):
    quit()

# Set the SNOPT options
trajopt.useSnoptSolver()
trajopt.setSolverOptions({'Iterations limit': 10000,
                        'Major feasibility tolerance': 1e-6,
                        'Major optimality tolerance': 1e-6,
                        'Scale option': 2})

slacks = [10, 1, 0.1, 0.01, 0.001, 0.]
for slack in slacks:
    print(f"Solving with complementarity slack: {slack}")
    # Increase the complementarity cost weight
    trajopt.const_slack = slack
    # Solve the problem
    result = trajopt.solve()
    utils.printProgramReport(result, trajopt.prog)
    # Save the results
    dir = os.path.join('data', 'slidingblock', 'collocation_slacktest',f'slack_{slack:.0e}')
    soln = trajopt.result_to_dict(result)
    os.makedirs(dir)
    file = os.path.join(dir, 'block_collocation_results.pkl')
    utils.save(file, soln)
    text = trajopt.generate_report(result)
    report = os.path.join(dir, 'block_collocation_report.txt')
    with open(report, "w") as file:
        file.write(text)
    # Make figures from the results
    xtraj, utraj, ftraj = trajopt.reconstruct_all_trajectories(result)[0:3]
    plant.plot_trajectories(xtraj, utraj, ftraj, samples=10000, show=False, savename=os.path.join(dir, 'BlockCollocation.png'))
    # Re-make the initial guess for the next iteration
    trajopt.initialize_from_previous(result)