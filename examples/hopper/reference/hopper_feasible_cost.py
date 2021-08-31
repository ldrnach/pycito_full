"""
Trajectory Optimization for the Footed Hopper with a running cost

Luke Drnach
August 25, 2021
"""

import numpy as np
from systems.hopper.hopper import Hopper
import trajopt.contactimplicit as ci
from pydrake.all import PiecewisePolynomial, SnoptSolver
import os
import utilities as utils
import matplotlib.pyplot as plt

# Save directory
savedir = os.path.join('examples','hopper','optimal_longfoot_cost')
os.makedirs(savedir)
# Create the footed hopper
hopper = Hopper()
hopper.Finalize()

# Initial and final states
base_0 = np.array([0., 1.5])
base_f = np.array([4., 1.5])
q_0, _  = hopper.standing_pose_ik(base_0)
no_vel = np.zeros((5,))
x_0 = np.concatenate((q_0, no_vel), axis=0)
x_f = x_0.copy()
x_f[:2] = base_f[:]

# Set up the optimization
N = 101
max_time = 3
min_time = 3

# Create optimization
context = hopper.multibody.CreateDefaultContext()
options = ci.OptimizationOptions()
options.useNonlinearComplementarityWithCost()
trajopt = ci.ContactImplicitDirectTranscription(hopper, context, num_time_samples=N, minimum_timestep=min_time/(N-1), maximum_timestep=max_time/(N-1), options=options)
trajopt.complementarity_cost_weight = [100, 100, 100]
# Boundary conditions
trajopt.add_state_constraint(knotpoint=0, value = x_0)
trajopt.add_state_constraint(knotpoint=N-1, value=x_f)

# Enforce equal timesteps
trajopt.add_equal_time_constraints()

# # Add running costs
R = 0.01*np.eye(3)
Q = np.diag([1, 10, 10, 100, 100, 1, 1, 1, 1, 1])
R = R/2
Q = Q/2

trajopt.add_quadratic_running_cost(R, np.zeros((3,)), vars=[trajopt.u], name='ControlCost')
trajopt.add_quadratic_running_cost(Q, x_f, vars=[trajopt.x], name='StateCost')

# Load the initial guess
# file = os.path.join('examples','hopper','feasible_scaled_moreiter','Slack_0E+00','trajoptresults.pkl')
# data = utils.load(file)
# trajopt.set_initial_guess(xtraj = data['state'], utraj=data['control'], ltraj=data['force'], jltraj=data['jointlimit'])

# Set the solver options
trajopt.setSolverOptions({'Iterations limit': 1000000,
                        'Major iterations limit': 5000,
                        'Minor iterations limit': 1000,
                        'Superbasics limit': 1500,
                        'Scale option': 2})
trajopt.enable_cost_display('figure')
# Set the force scaling
trajopt.force_scaling = 1
# Set the complementarity slack
# trajopt.const_slack = 0

# Solve the problem
result = trajopt.solve()
print(f"Successful? {result.is_success()}")
# Save the outputs
report = trajopt.generate_report(result)
reportfile = os.path.join(savedir, 'report.txt')
with open(reportfile, 'w') as file:
    file.write(report)
utils.save(os.path.join(savedir, 'trajoptresults.pkl'), trajopt.result_to_dict(result))
# Save the cost figure
trajopt.printer.save_and_close(os.path.join(savedir, 'CostsAndConstraints.png'))
# Plot and save the trajectories
xtraj, utraj, ftraj, jltraj, _ = trajopt.reconstruct_all_trajectories(result)
hopper.plot_trajectories(xtraj, utraj, ftraj, jltraj, show=False, savename=os.path.join(savedir, 'opt.png'))
plt.close('all')