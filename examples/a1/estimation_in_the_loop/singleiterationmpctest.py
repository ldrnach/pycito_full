"""
Run a single interation of MPC to tune SNOPT

Luke Drnach
June 28, 2022
"""
import os

from pycito import utilities as utils
from pycito.controller.optimization import OptimizationLogger


import a1_mpc_tools as mpctools
import flatterrain_mpc 

SOURCEDIR = os.path.join('examples','a1','estimation_in_the_loop','mpc','flatterrain_relaxed','3m')
SIMDATA = os.path.join(SOURCEDIR, 'simdata.pkl')
MPCLOGS = os.path.join(SOURCEDIR, 'mpclogs.pkl')

TARGET = os.path.join(SOURCEDIR, 'debug','linesearch_0.9_Superbasics_1e3')

INDEX = 20

# Make the controller and model
a1 = flatterrain_mpc.make_flatterrain_model()
reftraj = mpctools.get_reference_trajectory(flatterrain_mpc.SOURCE)
controller = mpctools.make_mpc_controller(reftraj, horizon=5)
controller.enable_cost_display(display='figure') 
# Set the pointer to a specific subproblem

mpclogs = OptimizationLogger.load(MPCLOGS)

t = mpclogs.logs[INDEX]['time']
x = mpclogs.logs[INDEX]['initial_state']

guess = mpclogs.guess_logs[INDEX]

# Create the MPC and set the initial guess
controller._cache = {'dx': guess['state'],
                    'du': guess['control'],
                    'dl': guess['force'],
                    'ds': guess['slack'],
                    'djl': guess['joint_limits']}
controller.create_mpc_program(t, x)

# Solve the problem
print(controller.complementarity_penalty)
controller.complementarity_penalty = controller.complementarity_schedule[0]
controller.setSolverOptions({'Linesearch tolerance': 0.9,
                            'Print file': os.path.join(TARGET, 'snopt.txt'),
                            'Superbasics limit': 1000})

guessreport = utils.printProgramInitialGuessReport(controller.prog, terminal=True)
if not os.path.exists(TARGET):
    os.makedirs(TARGET)
with open(os.path.join(TARGET, 'GuessReport.txt'), 'w') as file:
    file.write(guessreport)


result = controller.solve()
utils.printProgramReport(result, controller.prog, verbose=True, filename=os.path.join(TARGET, 'SolutionReport.txt'))



controller.printer.save_and_close(savename=os.path.join(TARGET, 'CostAndConstraints.png'))
