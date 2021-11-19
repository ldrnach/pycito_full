"""
Orthogonal Collocation for A1

Luke Drnach
November 16, 2021
"""

# Imports
import os
import numpy as np
import trajopt.contactimplicit as ci
from systems.A1.a1 import A1VirtualBase
import utilities as utils

def make_a1_walking_optimization(useipopt = False):
    # Create the block plant
    plant = A1VirtualBase()
    plant.Finalize()
    # Get the default context
    context = plant.multibody.CreateDefaultContext()
    options = ci.OrthogonalOptimizationOptions()
    options.useComplementarityWithCost()
    # Create a Contact Implicit OrthogonalCollocation
    N = 26
    max_time = 2
    min_time = 2

    trajopt = ci.ContactImplicitOrthogonalCollocation(plant, context, 
                                                    num_time_samples = N, 
                                                    minimum_timestep=min_time/(N-1), 
                                                    maximum_timestep=max_time/(N-1),
                                                    state_order=3,
                                                    options=options)
    #Additional constraints
    trajopt.add_zero_acceleration_boundary_condition()

    # Boundary conditions - Get from A1
    pose = plant.standing_pose()
    no_vel = np.zeros((plant.multibody.num_velocities(), ))
    x0 = np.concatenate((pose, no_vel), axis=0)
    xf = x0.copy()
    xf[0] = 1.
    Ntotal = trajopt.total_knots
    trajopt.add_state_constraint(knotpoint=0, value=x0)
    trajopt.add_state_constraint(knotpoint=Ntotal-1, value=xf)

    # Add cost functions
    R = np.eye(trajopt.u.shape[0])
    b = np.zeros((trajopt.u.shape[0], ))
    trajopt.add_quadratic_control_cost(R, b)

    state_weight = np.concatenate([10*np.ones((pose.shape[0],)), 100*np.ones((pose.shape[0],))])
    state_weight[2] = 1000
    Q = np.diag(state_weight)
    trajopt.add_quadratic_running_cost(Q, xf, vars=[trajopt.x], name='StateCost')
    
    if useipopt:
        # Set IPOPT options
        trajopt.useIpoptSolver()
        trajopt.setSolverOptions({'max_iter': 10000})
    else:
        # Set the SNOPT options
        trajopt.useSnoptSolver()
        trajopt.setSolverOptions({'Iterations limit': 1000000,
                                'Major iterations limit': 5000,
                                'Major feasibility tolerance': 1e-6,
                                'Major optimality tolerance': 1e-6,
                                'Scale option': 2})
    # Check the program
    print(f"Checking program")
    if not utils.CheckProgram(trajopt.prog):
        quit()
    return plant, trajopt, (x0, xf)

def set_default_initial_guess(trajopt, x0, xf):
    # Set the initial guess
    uinit = np.zeros(trajopt.u.shape)
    xinit = np.zeros(trajopt.x.shape)
    for n in range(0, xinit.shape[0]):
        xinit[n,:] = np.linspace(start=x0[n], stop=xf[n], num=trajopt.x.shape[1])
    linit = np.zeros(trajopt.l.shape)
    sinit = np.zeros(trajopt.var_slack.shape)
    ainit = np.zeros(trajopt.accel.shape)
    hinit = np.ones(trajopt.h.shape) * trajopt.maximum_timestep
    # Set the initial guess
    trajopt.set_initial_guess(xtraj=xinit, utraj=uinit, ltraj=linit, straj=sinit)
    trajopt.prog.SetInitialGuess(trajopt.accel, ainit)
    trajopt.prog.SetInitialGuess(trajopt.h, hinit)
    return trajopt

def initialize_from_file(trajopt, filename):
    data = utils.load(filename)
    trajopt.prog.SetInitialGuess(trajopt.h, data['timesteps'])
    trajopt.prog.SetInitialGuess(trajopt.x, data['state'])
    trajopt.prog.SetInitialGuess(trajopt.u, data['control'])
    trajopt.prog.SetInitialGuess(trajopt.l, data['force'])
    trajopt.prog.SetInitialGuess(trajopt.jl, data['jointlimit'])
    trajopt.prog.SetInitialGuess(trajopt.var_slack, data['slacks'])
    trajopt.prog.SetInitialGuess(trajopt.accel, data['acceleration'])
    return trajopt

def progressive_solve(trajopt, plant, weights, useipopt):

    # Save the report
    if useipopt:
        dir_base = os.path.join('data','a1_walking','collocation_ipopt')
    else:
        dir_base = os.path.join('data','a1_walking','collocatio')
        

    

    for weight in weights:
        print(f"Solving with complementarity weight: {weight}")
        # Increase the complementarity cost weight
        trajopt.complementarity_cost_weight = weight
        # Solve the problem
        result = trajopt.solve()
        utils.printProgramReport(result, trajopt.prog)
        # Save the results
        soln = trajopt.result_to_dict(result)
        dir = os.path.join(dir_base, f'weight_{weight}')
        os.makedirs(dir)
        file = os.path.join(dir, 'a1_walking_collocation_results.pkl')
        utils.save(file, soln)
        # Make figures from the results
        xtraj, utraj, ftraj, jltraj = trajopt.reconstruct_all_trajectories(result)[0:4]
        plant.plot_trajectories(xtraj, utraj, ftraj, jltraj, samples=10000, show=False, savename=os.path.join(dir, 'A1WalkingCollocation.png'))
        text = trajopt.generate_report(result)
        report = os.path.join(dir, 'a1_walking_collocation_report.txt')
        with open(report, "w") as file:
            file.write(text)
       
        # Re-make the initial guess for the next iteration
        trajopt.initialize_from_previous(result)

def main_solve():
    plant, trajopt, boundary = make_a1_walking_optimization()
    trajopt = set_default_initial_guess(trajopt, boundary[0], boundary[1])
    weights = [1, 10, 100, 1000, 10000] 
    progressive_solve(trajopt, plant, weights)

def main_solve_ipopt():
    plant, trajopt, boundary = make_a1_walking_optimization(useipopt=True)
    trajopt = set_default_initial_guess(trajopt, boundary[0], boundary[1])
    weights = [1, 10, 100, 1000, 10000] 
    progressive_solve(trajopt, plant, weights, useipopt=True)

def continue_solve():
    plant, trajopt, _ = make_a1_walking_optimization()
    trajopt = initialize_from_file(trajopt, os.path.join('data','a1_walking','collocation','weight_1','a1_walking_collocation_results.pkl'))
    weights = [10, 100, 1000, 10000] 
    progressive_solve(trajopt, plant, weights)

if __name__ == '__main__':
    main_solve_ipopt()