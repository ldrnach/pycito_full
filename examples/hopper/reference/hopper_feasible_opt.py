import numpy as np
from systems.hopper.hopper import Hopper
import trajopt.contactimplicit as ci
from pydrake.all import PiecewisePolynomial, SnoptSolver
import os
import utilities as utils
import matplotlib.pyplot as plt

def create_hopper():
    hopper = Hopper()
    hopper.Finalize()
    N = 101

    # Create boundary constraints
    base_0 = np.array([0., 1.5])
    base_f = np.array([4., 1.5])

    q_0, status = hopper.standing_pose_ik(base_0)
    #q_0 = hopper.standing_pose(base_0)
    no_vel = np.zeros((5,))
    x_0 = np.concatenate((q_0, no_vel), axis=0)
    x_f = x_0.copy()
    x_f[:2] = base_f[:]
    return hopper, N, x_0, x_f

def create_hopper_optimization(hopper, N, x_0, x_f):
    # Create the system
    context = hopper.multibody.CreateDefaultContext()
    options = ci.OptimizationOptions()
    options.useNonlinearComplementarityWithConstantSlack()

    # Create the optimization
    
    max_time = 3
    min_time = 3
    trajopt = ci.ContactImplicitDirectTranscription(hopper, context, num_time_samples=N, minimum_timestep=min_time/(N-1), maximum_timestep=max_time/(N-1), options=options)

    # Add boundary constraints
    trajopt.add_state_constraint(knotpoint=0, value=x_0)
    trajopt.add_state_constraint(knotpoint=trajopt.num_time_samples-1, value=x_f)

    trajopt.setSolverOptions({'Iterations limit': 100000,
                            'Major iterations limit': 5000,
                            'Minor iterations limit': 1000, 
                            'Superbasics limit': 1500,
                            'Scale option': 1,
                            'Elastic weight': 10**5})
    trajopt.enable_cost_display('figure')
    # Set variable scaling
    trajopt.force_scaling = 100
    # Require equal timesteps
    trajopt.add_equal_time_constraints()
    return trajopt

def create_initial_guess(N, x_0, x_f):
    # Create the initial guess
    t_init = np.linspace(0., 3., N)
    #x_init = np.linspace(x_0, x_f, N).transpose()
    x_init = np.zeros((10, N))
    x_init = PiecewisePolynomial.FirstOrderHold(t_init, x_init)

    u_init = PiecewisePolynomial.FirstOrderHold(t_init, np.zeros((3,N)))
    l_init = PiecewisePolynomial.FirstOrderHold(t_init, np.zeros((12,N)))
    jl_init = PiecewisePolynomial.FirstOrderHold(t_init, np.zeros((6,N)))
    #slack = PiecewisePolynomial.FirstOrderHold(t_init, np.zeros((12,N)))

    return x_init, u_init, l_init, jl_init

def main():
    hopper, N, x_0, x_f = create_hopper()
    x_init, u_init, l_init, jl_init = create_initial_guess(N, x_0, x_f)
    path = os.path.join('/workspaces', 'pyCITO','examples','hopper','feasible_longfoot_ncc')
    slacks = [10., 1.0, 0.1, 0.01, 0.001, 0.0001, 0.]

    breaks = x_init.get_segment_times()
    for slack in slacks:
        # Create a trajopt
        trajopt = create_hopper_optimization(hopper, N, x_0, x_f)
        # Make the output directory
        save_dir = os.path.join(path, f'Slack_{slack:.0E}')
        os.makedirs(save_dir)
        # Update the optimization
        trajopt.const_slack = slack
        trajopt.set_initial_guess(xtraj=x_init.vector_values(breaks), utraj=u_init.vector_values(breaks), ltraj=l_init.vector_values(breaks), jltraj=jl_init.vector_values(breaks))#, straj=sl_init.vector_values(breaks))
        # Solve the program
        print(f"Solving for slack {slack:.0E}")
        result = trajopt.solve()
        # Get the solution report
        report = trajopt.generate_report(result)
        # Save the results
        reportfile = os.path.join(save_dir, 'report.txt')
        with open(reportfile, 'w') as file:
            file.write(report)
        utils.save(os.path.join(save_dir, 'trajoptresults.pkl'), trajopt.result_to_dict(result))
        # Save the cost figure
        trajopt.printer.save_and_close(os.path.join(save_dir, 'CostsAndConstraints.png'))
        # Re-create the initial guess
        x_init, u_init, l_init, jl_init, _ = trajopt.reconstruct_all_trajectories(result)
        # Plot the trajectories
        hopper.plot_trajectories(x_init, u_init, l_init, jl_init, show=False, savename=os.path.join(save_dir, 'opt.png'))
        plt.close('all')


if __name__ == "__main__":
    main()
