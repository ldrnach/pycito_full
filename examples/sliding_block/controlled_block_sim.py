"""
Comparison of open and closed loop simulations using the sliding block

Luke Drnach
February 8, 2022
"""
import os

from pydrake.all import PiecewisePolynomial as pp

from pycito.systems.block.block import Block
from pycito.systems.simulator import Simulator
import pycito.controller.mpc as mpc
import pycito.utilities as utils
import pycito.systems.terrain as terrain

# Globals
SOURCE = os.path.join('data','slidingblock','block_trajopt.pkl')
SAVEDIR = os.path.join('examples','sliding_block','simulations')
FILENAME = 'simdata.pkl'

# Friction models
def low_friction(x):
    if x[0] < 2.0 or x[0] > 4.0:
        return 0.5
    else:
        return 0.1

def high_friction(x):
    if x[0] < 2.0 or x[0] > 4.0:
        return 0.5
    else:
        return 0.9

def plot_sim_results(plant, t, x, u, f, savedir, vis=True):
    """Plot the block trajectories"""
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    xtraj = pp.FirstOrderHold(t, x)
    utraj = pp.ZeroOrderHold(t, u)
    ftraj = pp.ZeroOrderHold(t, f)
    plant.plot_trajectories(xtraj, utraj, ftraj, show=False, savename=os.path.join(savedir, 'sim.png'))
    # Visualize in meshcat
    if vis:
        plant.visualize(xtraj)

def run_simulation(plant, controller, initial_state, duration, savedir=None):
    #Check if the target directory exists
    if savedir is not None and not os.path.exists(savedir):
        os.makedirs(savedir)
    # Create and run the simulation
    sim = Simulator(plant, controller)
    tsim, xsim, usim, fsim = sim.simulate(initial_state, duration)
    # Save the results
    plot_sim_results(plant, tsim, xsim, usim, fsim, savedir=savedir)
    # Save the data
    simdata = {'time': tsim,
                'state': xsim,
                'control': usim,
                'force': fsim}
    if savedir is not None:
        utils.save(os.path.join(savedir, FILENAME), simdata)
        print(f"Simulation results saved to {os.path.join(savedir, FILENAME)}")
    return simdata

def get_block_mpc_controller():
    # Create the 'reference' model
    block = Block()
    block.Finalize()
    # Load the reference trajectory
    reftraj = mpc.LinearizedContactTrajectory.load(block, SOURCE)
    # Make the controller
    controller = mpc.LinearContactMPC(reftraj, horizon = 10)
    controller.useSnoptSolver()
    controller.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    return controller

def get_block_open_loop_controller():
    # Create the 'reference' model
    block = Block()
    block.Finalize()
    # Load the reference trajectory
    reftraj = mpc.ReferenceTrajectory.load(block, SOURCE)

def flatterrain_sim():
    # Get the reference controller
    controller = get_block_mpc_controller()
    open_loop = get_block_open_loop_controller()
    # Create the 'true model' for the simulator
    block = Block()
    block.Finalize()
    # Initial state
    x0 = controller.lintraj.getState(0)
    # Run the open-loop simulation
    print("Running low friction open loop simulation")
    run_simulation(block, open_loop, x0, duration = 1, savedir=os.path.join(SAVEDIR, 'flatterrain', 'openloop'))
    # Run the mpc simulation
    print("Running low friction MPC simulation")
    run_simulation(block, controller, x0, duration = 1, savedir=os.path.join(SAVEDIR, 'flatterrain', 'mpc'))

def lowfriction_sim():
    """Run open and closed loop simulations on terrain with low friction"""
    # Get the reference controller
    controller = get_block_mpc_controller()
    open_loop = get_block_open_loop_controller()
    # Create the 'true model' for the simulator
    lowfriction_terrain = terrain.VariableFrictionFlatTerrain(height=0, fric_func=low_friction)
    block = Block(terrain = lowfriction_terrain)
    block.Finalize()
    # Initial state
    x0 = controller.lintraj.getState(0)
    # Run the open-loop simulation
    print("Running low friction open loop simulation")
    run_simulation(block, open_loop, x0, duration = 1, savedir=os.path.join(SAVEDIR, 'lowfriction', 'openloop'))
    # Run the mpc simulation
    print("Running low friction MPC simulation")
    run_simulation(block, controller, x0, duration = 1, savedir=os.path.join(SAVEDIR, 'lowfriction', 'mpc'))

def highfriction_sim():
    """Run open and closed loop simulations on terrian with high friction"""
    # Create the 'reference model' for the controller
    controller = get_block_mpc_controller()
    open_loop = get_block_open_loop_controller()
    # Create the 'true model' for the simulator
    highfriction_terrain = terrain.VariableFrictionFlatTerrain(height=0, fric_func=high_friction)
    block = Block(terrain = highfriction_terrain)
    block.Finalize()
    # Initial state
    x0 = controller.lintraj.getState(0)
    # Run open loop simulation
    print('Running high friction open loop simulation')
    run_simulation(block, open_loop, x0, duration = 1, savedir = os.path.join(SAVEDIR, 'highfriction', 'openloop'))
    # Run mpc simulation
    print('Running high friction MPC simulation')
    run_simulation(block, controller, x0, duration = 1, savedir = os.path.join(SAVEDIR, 'highfriction','mpc'))

def steppedterrain_sim():
    """Run open and closed loop simulations on terrain with a step in it"""
    # Create the 'reference model' for the controller
    controller = get_block_mpc_controller()
    open_loop = get_block_open_loop_controller()
    # Create the 'true model' for the simulator
    stepped_terrain = terrain.StepTerrain(step_height = -0.5, step_location = 2.5)
    block = Block(terrain = stepped_terrain)
    block.Finalize()
    # Initial state
    x0 = controller.lintraj.getState(0)
    # Run open loop simulation
    print('Running high friction open loop simulation')
    run_simulation(block, open_loop, x0, duration = 1, savedir = os.path.join(SAVEDIR, 'steppedterrain', 'openloop'))
    # Run mpc simulation
    print('Running high friction MPC simulation')
    run_simulation(block, controller, x0, duration = 1, savedir = os.path.join(SAVEDIR, 'steppedterrain','mpc'))

if __name__ == '__main__':
    flatterrain_sim()
    lowfriction_sim()
    highfriction_sim()
    steppedterrain_sim()

