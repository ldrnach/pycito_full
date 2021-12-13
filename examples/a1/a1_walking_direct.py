"""
A1 Trajectory Optimization

Luke Drnach
November 2021
"""

# Imports
import os
import numpy as np
import trajopt.contactimplicit as ci
from systems.A1.a1 import A1VirtualBase
import utilities as utils
from matplotlib import pyplot as plt


def make_a1_walking_optimization():
    # Create the block plant
    plant = A1VirtualBase()
    plant.Finalize()
    plant.terrain.friction = 1.0
    # Get the default context
    context = plant.multibody.CreateDefaultContext()
    options = ci.OptimizationOptions()
    options.useNonlinearComplementarityWithCost()
    # Create a Contact Implicit OrthogonalCollocation
    N = 51
    max_time = 2
    min_time = 2

    trajopt = ci.ContactImplicitDirectTranscription(plant, context, 
                                                    num_time_samples = N, 
                                                    minimum_timestep=min_time/(N-1), 
                                                    maximum_timestep=max_time/(N-1),
                                                    options=options)
    # Boundary conditions - Get from A1
    pose = plant.standing_pose()
    pose, _ = plant.standing_pose_ik(base_pose=pose[0:6], guess=pose)
    no_vel = np.zeros((plant.multibody.num_velocities(), ))
    x0 = np.concatenate((pose, no_vel), axis=0)
    xf = x0.copy()
    xf[0] = 1.
    trajopt.add_state_constraint(knotpoint=0, value=x0)
    trajopt.add_state_constraint(knotpoint=N-1, value=xf)
    trajopt.add_equal_time_constraints()
    # Add cost functions
    R = 100*np.eye(trajopt.u.shape[0])
    b = np.zeros((trajopt.u.shape[0], ))
    trajopt.add_quadratic_running_cost(R, b, vars=[trajopt.u], name='ControlCost')

    state_weight = np.concatenate([10*np.ones((pose.shape[0],)), 100*np.ones((pose.shape[0],))])
    Q = np.diag(state_weight)
    trajopt.add_quadratic_running_cost(Q, xf, vars=[trajopt.x], name='StateCost')
    # Set the SNOPT options
    trajopt.useSnoptSolver()
    trajopt.setSolverOptions({'Iterations limit': 1000000,
                            'Major iterations limit': 5000,
                            'Major feasibility tolerance': 1e-6,
                            'Major optimality tolerance': 1e-6,
                            'Scale option': 2})
    trajopt.enable_cost_display(display='terminal')
    # Check the program
    print(f"Checking program")
    if not utils.CheckProgram(trajopt.prog):
        quit()
    return plant, trajopt, (x0, xf)

def initialize_from_file(trajopt, init_file=None):
    if init_file and os.path.exists(init_file):
        guess = utils.load(init_file)
        trajopt.set_initial_guess(xtraj=guess['state'], utraj=guess['control'], ltraj=guess['force'], jltraj=guess['jointlimit'])
    return trajopt

def progressive_solve(trajopt, plant, weights):

    # Save the report
    dir_base = os.path.join('data','a1_walking','walking_warmstart_51_feasible')
    
    for weight in weights:
        print(f"Solving with complementarity weight: {weight}")
        # Increase the complementarity cost weight
        trajopt.complementarity_cost_weight = weight
        # Solve the problem
        result = trajopt.solve()
        utils.printProgramReport(result, trajopt.prog)
        # Save the results
        soln = trajopt.result_to_dict(result)
        dir = os.path.join(dir_base, f'weight_{weight:.0e}')
        os.makedirs(dir)
        file = os.path.join(dir, 'trajoptresults.pkl')
        utils.save(file, soln)
        # Make figures from the results
        xtraj, utraj, ftraj, jltraj = trajopt.reconstruct_all_trajectories(result)[0:4]
        plant.plot_trajectories(xtraj, utraj, ftraj, jltraj, show=False, savename=os.path.join(dir, 'A1WalkingWarmstart51.png'))
        text = trajopt.generate_report(result)
        report = os.path.join(dir, 'report.txt')
        with open(report, "w") as file:
            file.write(text)
        # Re-make the initial guess for the next iteration
        trajopt.initialize_from_previous(result)
        plt.close('all')

def main(init_file=None):
    a1, trajopt, boundary = make_a1_walking_optimization()
    trajopt = initialize_from_file(trajopt, init_file)
    #slacks = [10, 1, 0.1, 0.01, 0.001, 0]
    weights = [1, 10, 100, 1000, 10000]
    progressive_solve(trajopt, a1, weights)

if __name__=='__main__':
    warmstart = os.path.join('data','a1','warmstarts','staticwalking_51.pkl')
    main(warmstart)