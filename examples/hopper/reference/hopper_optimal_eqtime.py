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
from examples.hopper.hopper_feasible_opt import create_hopper, create_hopper_optimization

# Save directory
savedir = os.path.join('examples','hopper','optimal_eqtime')
os.makedirs(savedir)
# Create the footed hopper
hopper, N, x_0, x_f = create_hopper()
trajopt = create_hopper_optimization(hopper, N, x_0, x_f)
# Add running costs
R = 0.01*np.eye(3)
Q = np.diag([1, 10, 10, 100, 100, 1, 1, 1, 1, 1])

trajopt.add_quadratic_running_cost(R, np.zeros((3,)), vars=[trajopt.u], name='ControlCost')
trajopt.add_quadratic_running_cost(Q, x_f, vars=[trajopt.x], name='StateCost')

# Load the initial guess
file = os.path.join('examples','hopper','feasible_eqtime','Slack2_0E+00','trajoptresults.pkl')
data = utils.load(file)
trajopt.set_initial_guess(xtraj = data['state'], utraj=data['control'], ltraj=data['force'], jltraj=data['jointlimit'])

# Set the solver options
trajopt.setSolverOptions({'Iterations limit': 100000,
                        'Major iterations limit': 5000,
                        'Minor iterations limit': 1000,
                        'Superbasics limit': 1500,
                        'Scale option': 2,
                        'Elastic weight': 10**5})
# Set the force scaling
trajopt.force_scaling = 100
# Set the complementarity slack
trajopt.const_slack = 0

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