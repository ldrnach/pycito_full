import utilities as utils
import os
from examples.hopper.hopper_feasible_opt import create_hopper, create_hopper_optimization
import matplotlib.pyplot as plt

# Load the warmstart data
file = os.path.join('examples','hopper','feasible_eqtime','Slack_1E-04','trajoptresults.pkl')
data = utils.load(file)
# Create the problem
hopper, N, x_0, x_f = create_hopper()
trajopt = create_hopper_optimization(hopper, N, x_0, x_f)
# Set the slack 
trajopt.const_slack = 0.
# Update the initial guess
trajopt.set_initial_guess(xtraj=data['state'], utraj=data['control'],ltraj=data['force'], jltraj=data['jointlimit'])
# Update the elastic weight
trajopt.setSolverOptions({'Elastic weight': 10**5})
# Solve the optimization
savedir = os.path.join('examples','hopper','feasible_eqtime','Slack2_0E+00')
os.makedirs(savedir)
result = trajopt.solve()
print(f"Optimization successful? {result.is_success()}")
# Get the solution report
report = trajopt.generate_report(result)
# Save the results
reportfile = os.path.join(savedir, 'report.txt')
with open(reportfile, 'w') as file:
    file.write(report)
utils.save(os.path.join(savedir, 'trajoptresults.pkl'), trajopt.result_to_dict(result))
# Save the cost figure
trajopt.printer.save_and_close(os.path.join(savedir, 'CostsAndConstraints.png'))
# Re-create the initial guess
x_init, u_init, l_init, jl_init, _ = trajopt.reconstruct_all_trajectories(result)
# Plot the trajectories
hopper.plot_trajectories(x_init, u_init, l_init, jl_init, show=False, savename=os.path.join(savedir, 'opt.png'))
plt.close('all')
