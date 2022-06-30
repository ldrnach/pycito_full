"""
Run two steps of MPC using the old basis from the previous solve in the new solve

Luke Drnach
"""

import os, time
import numpy as np
from pycito import utilities as utils
from pycito.controller.optimization import OptimizationLogger

import a1_mpc_tools as mpctools
import flatterrain_mpc

SOURCEDIR = os.path.join('examples','a1','estimation_in_the_loop','mpc','flatterrain_relaxed','3m')
SIMDATA = os.path.join(SOURCEDIR, 'simdata.pkl')
MPCLOGS = os.path.join(SOURCEDIR, 'mpclogs.pkl')

TARGET = os.path.join(SOURCEDIR, 'debug','baditer')


INDEX = 29

# Make controller and model
a1 = flatterrain_mpc.make_flatterrain_model()
reftraj = mpctools.get_reference_trajectory(flatterrain_mpc.SOURCE)
controller = mpctools.make_mpc_controller(reftraj, horizon=5)
#controller.enable_cost_display(display='figure') 

# Set the pointer to a specific subproblem
mpclogs = OptimizationLogger.load(MPCLOGS)

t = mpclogs.logs[INDEX]['time']
x = mpclogs.logs[INDEX]['initial_state']

# Create the MPC
controller.create_mpc_program(t, x)
controller.setSolverOptions({'Scale option': 1,
                            'Superbasics limit':1000,
                            'New basis file': 3,
                            'Linesearch tolerance':0.1,
                            'Major feasibility tolerance':1e-5,
                            'Major optimality tolerance':1e-5,
                            'Major step limit': 0.5})

# Run the 'coldstart' run
guessreport = utils.printProgramInitialGuessReport(controller.prog, terminal=True)
target_one = os.path.join(TARGET, 'firstrun')
if not os.path.exists(target_one):
    os.makedirs(target_one)
with open(os.path.join(target_one, 'GuessReport.txt'), 'w') as file:
    file.write(guessreport)
start = time.perf_counter()
result = controller.progressive_solve()
elapsed = time.perf_counter() - start
utils.printProgramReport(result, controller.prog, verbose=True, filename=os.path.join(target_one,'SolutionReport.txt'))
##controller.printer.save_and_clear(savename=os.path.join(target_one, 'CostAndConstraints.png'))

# Run the 'warmstart' run
t, x = mpclogs.logs[INDEX+1]['time'], mpclogs.logs[INDEX+1]['initial_state']
controller.create_mpc_program(t, x)
controller.setSolverOptions({'Old basis file': 3})
controller.complementarity_schedule = [1e-4]
dvars = controller.prog.decision_variables()
controller.prog.SetInitialGuess(dvars, result.GetSolution(dvars))
guessreport = utils.printProgramInitialGuessReport(controller.prog, terminal=True)
target_two = os.path.join(TARGET, 'secondrun')
if not os.path.exists(target_two):
    os.makedirs(target_two)
with open(os.path.join(target_two, 'GuessReport.txt'), 'w') as file:
    file.write(guessreport)

start = time.perf_counter()
result = controller.solve()
elapsed_basis = time.perf_counter() - start

utils.printProgramReport(result, controller.prog, verbose=True, filename=os.path.join(target_two,'SolutionReport.txt'))
#controller.printer.save_and_close(savename=os.path.join(target_two, 'CostAndConstraints.png'))
print(f"Time to solve the original problem : {elapsed:0.2f}")
print(f"Time to solve the second problem, with old basis: {elapsed_basis:0.2f}")