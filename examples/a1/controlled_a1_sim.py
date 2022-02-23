"""
Comparison of open and closed loop simulations using A1

Luke Drnach
February 9, 2022
"""

import os
import numpy as np

from pydrake.all import PiecewisePolynomial as pp

from pycito.systems.A1.a1 import A1VirtualBase
from pycito.systems.simulator import Simulator
import pycito.controller.mpc as mpc
import pycito.utilities as utils

SOURCE = os.path.join('examples','a1','foot_tracking_gait','first_step','weight_1e+03','trajoptresults.pkl')
SAVEDIR = os.path.join('examples','a1','simulations')
FILENAME = 'simdata.pkl'

def visualize_controlled_trajectory():
    plant = A1VirtualBase()
    plant.Finalize()
    source = os.path.join('examples','a1','simulations','closed_loop','simdata.pkl')
    data = utils.load(utils.FindResource(source))
    xtraj = pp.FirstOrderHold(data['time'], data['state'])
    plant.visualize(xtraj)

def visualize_open_loop_trajectory():
    plant = A1VirtualBase()
    plant.Finalize()
    source = os.path.join('examples','a1','simulations','open_loop','simdata.pkl')
    data = utils.load(utils.FindResource(source))
    xtraj = pp.FirstOrderHold(data['time'][:11], data['state'][:, :11])
    plant.visualize(xtraj)

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

def run_simulation(plant, controller, initial_state, duration, savedir=None, vis=False):
    #Check if the target directory exists
    if savedir is not None and not os.path.exists(savedir):
        os.makedirs(savedir)
    # Create and run the simulation
    sim = Simulator(plant, controller)
    tsim, xsim, usim, fsim, status = sim.simulate(initial_state, duration)
    if ~status:
        print(f"Simulation failed at timestep {tsim[-1]}")
    # Save the results
    plot_sim_results(plant, tsim, xsim, usim, fsim, savedir=savedir, vis=vis)
    # Save the data
    simdata = {'time': tsim,
                'state': xsim,
                'control': usim,
                'force': fsim,
                'succees': status}
    if savedir is not None:
        utils.save(os.path.join(savedir, FILENAME), simdata)
        print(f"Simulation results saved to {os.path.join(savedir, FILENAME)}")
    return simdata

def get_a1_mpc_controller(source = SOURCE, horizon = 5):
    # Create the reference model
    plant = A1VirtualBase()
    plant.terrain.friction = 1.0
    plant.Finalize()
    # Load the reference trajectory
    reftraj = mpc.LinearizedContactTrajectory.load(plant, source)
    # Make the controller
    controller = mpc.LinearContactMPC(reftraj, horizon = 5)
    controller.useSnoptSolver()
    controller.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    # Set the cost terms
    controller.complementaritycost = 1e3
    controller.statecost = np.diag([1e2] * plant.multibody.num_positions() + [1e-2] * plant.multibody.num_velocities())
    controller.controlcost = 1e-2 * np.eye(controller.control_dim)
    controller.forcecost = 1e-2 * np.eye(controller.force_dim)
    controller.slackcost = 1e-2 * np.eye(controller.slack_dim)
    controller.limitcost = 1e-2 * np.eye(reftraj._jlimit.shape[0])

    return controller

def flatground_walking_sim():
    source = os.path.join('examples','a1','foot_tracking_fast_shift','10step_plots','combinedresults.pkl')
    savedir = os.path.join('examples','a1','simulations','flatground_walking_fast')
    # Get the reference controller
    controller = get_a1_mpc_controller(source=source)
    # Create the 'true model' for simulation
    a1 = A1VirtualBase()
    a1.terrain.friction = 1.0
    a1.Finalize()
    # Initial state
    x0 = controller.lintraj.getState(0)
    T = controller.lintraj._time[-1]
    # Run the closed loop simulation
    print(f"Running flat ground walking A1 simulations")
    run_simulation(a1, controller, x0, T, savedir = None, vis=False)

def main():
    # Data source
    source = os.path.join('examples','a1','foot_tracking_gait','first_step','weight_1e+03','trajoptresults.pkl')
    savedir = os.path.join('examples','a1','simulations')
    # Make the plant model
    plant = A1VirtualBase()
    plant.Finalize()
    # Create the controller
    reftraj = mpc.LinearizedContactTrajectory.load(plant, source)
    controller = mpc.LinearContactMPC(reftraj, horizon=10)
    # Set the cost weights
    controller.complementaritycost = 1e3
    controller.statecost = np.diag([1e2] * plant.multibody.num_positions() + [1e-2] * plant.multibody.num_velocities())
    controller.controlcost = np.diag([1e-2] * plant.multibody.num_actuators())
    controller.forcecost = np.diag([1e-2] * (plant.num_contacts() + plant.num_friction()))
    controller.slackcost = np.diag([1e-2] * plant.num_contacts())
    controller.limitcost = np.diag([1e-2] * reftraj._jlimit.shape[0])
    # Get the open loop control
    utraj = pp.ZeroOrderHold(reftraj._time, reftraj._control)
    # Setup solver options
    controller.useSnoptSolver()
    controller.setSolverOptions({'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6})
    # Create the simulator
    opensim = Simulator.OpenLoop(plant, utraj)
    closedsim = Simulator.ClosedLoop(plant, controller)
    # Run the simulations
    initial_state = reftraj.getState(0)
    print(f"Running open loop simulation")
    topen, xopen, uopen, fopen = opensim.simulate(initial_state, duration=0.5)
    print(f"Running closed loop simulation")
    tcl, xcl, ucl, fcl = closedsim.simulate(initial_state, duration=0.5)
    # Plot the results
    plot_sim_results(plant, topen, xopen, uopen, fopen, savedir=os.path.join(savedir, 'open_loop'), vis=False)
    plot_sim_results(plant, tcl, xcl, ucl, fcl, savedir=os.path.join(savedir, 'closed_loop'), vis=False)
    # Save the data
    opendata = {'time': topen,
                'state': xopen,
                'control': uopen,
                'force': fopen}
    utils.save(os.path.join(savedir, 'open_loop','simdata.pkl'), opendata)
    closeddata = {'time': tcl,
                'state': xcl,
                'control': ucl,
                'force': fcl}
    utils.save(os.path.join(savedir, 'closed_loop','simdata.pkl'), closeddata)
    
def debug_a1_controller():
    source = os.path.join('examples','a1','foot_tracking_gait','first_step','weight_1e+03','trajoptresults.pkl')
    plant = A1VirtualBase()
    plant.Finalize()
    reftraj = mpc.LinearizedContactTrajectory.load(plant, source)
    controller = mpc.LinearContactMPC(reftraj, horizon=10)
    # Set the cost weights
    controller.complementaritycost = 1e3
    controller.statecost = np.diag([1e2] * plant.multibody.num_positions() + [1e-2] * plant.multibody.num_velocities())
    controller.controlcost = np.diag([1e-2] * plant.multibody.num_actuators())
    controller.forcecost = np.diag([1e-2] * (plant.num_contacts() + plant.num_friction()))
    controller.slackcost = np.diag([1e-2] * plant.num_contacts())
    controller.limitcost = np.diag([1e-2] * reftraj._jlimit.shape[0])
    # Create the MPC program at time instance 0.1
    controller.create_mpc_program(t=0.10, x0 = reftraj.getState(reftraj.getTimeIndex(0.10)))
    print("Program created successfully")
    # Check the constraints in the program
    for cstr in controller.prog.GetAllConstraints():
        lb = cstr.evaluator().lower_bound()
        if np.any(np.isnan(lb)):
            print(f"Constraint {cstr.evaluator().get_description()} has NAN in the lower bound")
        ub = cstr.evaluator().upper_bound()
        if np.any(np.isnan(ub)):
            print(f"Constraint {cstr.evaluator().get_description()} has NAN in the upper bound")


if __name__ == '__main__':
    #main()
    #visualize_open_loop_trajectory()
    #debug_a1_controller()
    flatground_walking_sim()