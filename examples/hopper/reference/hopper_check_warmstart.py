"""


"""
import utilities as utils
import os
from examples.hopper.hopper_feasible_opt import create_hopper, create_hopper_optimization

# Load the data file
file = os.path.join('examples','hopper','feasible_eqtime','Slack2_0E+00','trajoptresults.pkl')
data = utils.load(file)
# Re-create the trajopt that generated this data
hopper, N, x_0, x_f = create_hopper()
trajopt = create_hopper_optimization(hopper, N, x_0, x_f)
# Set the slack variable
trajopt.const_slack = 0.
# Set the initial guess
trajopt.set_initial_guess(xtraj=data['state'], utraj=data['control'], ltraj=data['force'], jltraj=data['jointlimit'])
# Check the constraint violations
trajopt.enable_cost_display('terminal')
# Get all decision variables and the initial guess values
dvars = trajopt.prog.decision_variables()
dvals = trajopt.prog.GetInitialGuess(dvars)
print(f"Costs and Constraints at the initial guess for slack = {trajopt.const_slack:.0E}")
trajopt.printer(dvals)

# Change the slack, and calculate again
# trajopt.const_slack = 1e-6
# print(f"Costs and constraints at the initial guess for slack = {trajopt.const_slack:.0E}")
# trajopt.printer(dvals)