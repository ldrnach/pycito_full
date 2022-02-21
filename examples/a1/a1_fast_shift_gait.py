import numpy as np
import os, copy

import pycito.utilities as utils
from a1_foot_tracking_opt import GaitGenerator, GaitType, A1FootTrackingCost, add_base_tracking
import a1trajopttools as opttools

def solve_forces(a1, qtraj):
    u = np.zeros((a1.multibody.num_actuators(), qtraj.shape[1]))
    fN = np.zeros((4, qtraj.shape[1]))
    for n in range(qtraj.shape[1]):
        u[:, n], fN[:, n] = a1.static_controller(qtraj[:, n])
    return u, fN

def setup_foot_tracking_gait(a1, foot_ref, base_ref, duration, warmstart, options=None):
    """Generically setup the foot tracking optimization"""
    N = warmstart['state'].shape[1]
    nQ = a1.multibody.num_positions()
    # Setup the trajectory optimization
    trajopt = opttools.make_a1_trajopt_linearcost(a1, N, [duration, duration])
    # Add boundary constraints
    trajopt = opttools.add_boundary_constraints(trajopt, warmstart['state'][:, 0], warmstart['state'][:, -1])
    # Set the initial guess
    trajopt.set_initial_guess(xtraj=warmstart['state'], utraj=warmstart['control'], ltraj=warmstart['force'], jltraj=warmstart['jointlimit'],straj=warmstart['slacks'])
    # Require foot tracking                           
    foot_cost = A1FootTrackingCost(a1, weight=1e4, traj=foot_ref)
    trajopt.prog = foot_cost.addToProgram(trajopt.prog, qvars=trajopt.x[:nQ,:])
    # Require base tracking
    trajopt = add_base_tracking(trajopt, base_ref, weight=1e2)
    # Set a small cost on control
    trajopt = opttools.add_control_cost(trajopt, weight=1e-2)
    # Add small cost on force
    trajopt = opttools.add_force_cost(trajopt, weight=1e-2)
    # Add small cost on control difference
    trajopt = opttools.add_control_difference_cost(trajopt, weight=1e-2)
    # Add small cost on force difference
    trajopt = opttools.add_force_difference_cost(trajopt, weight=1e-2)
    # Update the solver options
    if options:
        trajopt.setSolverOptions(options)
    return trajopt

def a1_fast_shifted_steps(a1, numsteps=1):
    generator = GaitGenerator(a1)
    #Starting configuration
    q0 = a1.standing_pose()
    q0, _ = a1.standing_pose_ik(base_pose = q0[:6], guess=q0)
    # Make the gait types
    gaits = [GaitType(stride_length=0.1, duration = 0.125), 
            GaitType(stride_length=0.2, duration=0.25),
            GaitType(stride_length=0.2, duration=0.25),
            GaitType(stride_length=0.1, duration = 0.125)]
    gaits[1].reversed = True
    gaits[3].reversed = True
    for gait in gaits:
        gait.use_quartic()
    samples = [13] + [26]*(2*numsteps) + [13]
    start, step, stop = gaits[0], gaits[1:3], gaits[3]
    for _ in range(numsteps-1):
        step.extend(step)
    gaits = [start] + step + [stop]
    # Make the gait trajectories
    step = []
    base = []
    q = []
    for gait, sample in zip(gaits, samples):
        step.append(generator.make_foot_stride_trajectory(q0, gait, sample))
        base.append(generator.make_base_trajectory(q0, step[-1]))
        q.append(generator.make_configuration_profile(step[-1], base[-1]))
        q0 = q[-1][:, -1]

    return step, base, q

def join_foot_trajectories(feet):
    """Join foot trajectories, knowing the endpoints of successive trajectories match"""
    joined_feet = feet.pop(0)
    for traj in feet:
        for k, foot in enumerate(traj):
            joined_feet[k] = np.concatenate([joined_feet[k], foot[:, 1:]], axis=1)

    return joined_feet

def join_configuration_trajectories(qtraj):
    """Join the configuration trajectories, knowing the endpoints of successive trajectories match"""
    for n in range(1, len(qtraj)):
        qtraj[n] = qtraj[n][:, 1:]
    return np.concatenate(qtraj, axis=1)

def create_warmstart(a1, q):
    warmstart = {'state': None,
                'control': None,
                'force': None,
                'jointlimit': None,
                'slacks': None}
    warmstart['control'], fN = solve_forces(a1, q)
    warmstart['force'] = np.zeros((2*a1.num_contacts() + a1.num_friction(), fN.shape[1]))
    warmstart['force'][:fN.shape[0], :] = fN
    warmstart['state'] = np.concatenate([q, np.zeros((a1.multibody.num_velocities(), q.shape[1]))], axis=0)
    return warmstart

def a1_first_step_fast_optimization():
    # Get the footstep parameters
    savedir = os.path.join('examples','a1','foot_tracking_fast_shift','firststep')
    a1 = opttools.make_a1()
    feet, _, qtraj = a1_fast_shifted_steps(a1)
    foot, q = feet[1], qtraj[1]
    duration = 0.25
    # Create the warmstart dictionary
    warmstart = create_warmstart(a1, q)
    # Create the trajopt
    trajopt = setup_foot_tracking_gait(a1, foot, q[:6, :], duration, warmstart)
    # Solve the problem
    weights = [1, 1e1, 1e2, 1e3]
    opttools.progressive_solve(trajopt, weights, savedir)

def a1_second_step_fast_optimization():
    # Get the footstep parameters
    savedir = os.path.join('examples','a1','foot_tracking_fast_shift','secondstep')
    a1 = opttools.make_a1()
    feet, _, qtraj = a1_fast_shifted_steps(a1)
    duration = 0.25
    foot, q = feet[2], qtraj[2]
    # Generate the warmstart
    warmstart = create_warmstart(a1, q)
    # Create the trajopt
    trajopt = setup_foot_tracking_gait(a1, foot, q[:6,:], duration, warmstart)
    # Add continuity constraints
    sourcedir = os.path.join('examples','a1','foot_tracking_fast_shift','firststep','weight_1e+03','trajoptresults.pkl')
    if os.path.exists(sourcedir):
        firststep = utils.load(utils.FindResource(sourcedir))
        # Add continuity constraints on forces and controls
        trajopt.prog.AddLinearEqualityConstraint(Aeq = np.eye(trajopt.l.shape[0]), beq=firststep['force'][:, -1], vars=trajopt.l[:, 0]).evaluator().set_description('Force Continuity')
        trajopt.prog.AddLinearEqualityConstraint(Aeq = np.eye(trajopt.u.shape[0]), beq=firststep['control'][:, -1], vars=trajopt.u[:, 0]).evaluator().set_description('Control Continuity')
        trajopt.prog.AddLinearEqualityConstraint(Aeq = np.eye(trajopt.jl.shape[0]), beq=firststep['jointlimit'][:, -1], vars=trajopt.jl[:, 0]).evaluator().set_description('JLimit Continuity')
        # Add periodicity constraints on forces and controls
        trajopt.prog.AddLinearEqualityConstraint(Aeq = np.eye(trajopt.l.shape[0]), beq=firststep['force'][:, 0], vars=trajopt.l[:,-1]).evaluator().set_description('Force Periodicity')
        trajopt.prog.AddLinearEqualityConstraint(Aeq = np.eye(trajopt.u.shape[0]), beq=firststep['control'][:, 0], vars=trajopt.u[:, -1]).evaluator().set_description('Control Periodicity')
        trajopt.prog.AddLinearEqualityConstraint(Aeq = np.eye(trajopt.jl.shape[0]), beq=firststep['jointlimit'][:, 0], vars=trajopt.jl[:, -1]).evaluator().set_description('JLimit Periodicity')
    else:
        print(f"Failed to add continuity and periodicity constraints")
    # Solve the problem
    weights = [1, 1e1, 1e2, 1e3]
    opttools.progressive_solve(trajopt, weights, savedir) 

def periodicity_parameters(dim):
    return np.concatenate([np.eye(dim), -np.eye(dim)], axis=1), np.zeros((dim, ))

def join_backward_euler_trajectories(dataset):
    """
    Join trajectories, ensuring dynamic consistency with respect to Backward Euler Dynamics
    """
    fulldata = dataset.pop(0)
    for data in dataset:
        # First, increment time appropriately
        t_new = data['time'][1:] + fulldata['time'][-1]
        fulldata['time'] = np.concatenate([fulldata['time'], t_new], axis=0)
        # Next, reset the base positions - translate the base
        data['state'][:3, :] += fulldata['state'][:3, -1:] - data['state'][:3, :1]
        # Join the state data
        fulldata['state'] = np.concatenate([fulldata['state'][:, :-1], data['state'][:, :]], axis=1)
        # Join the control torques - use the first control from the next segment
        fulldata['control'] = np.concatenate([fulldata['control'][:, :-1], data['control']], axis=1)
        # Join the reaction and joint limit forces appropriately - use the last force set from the previous segment
        fulldata['force'] = np.concatenate([fulldata['force'], data['force'][:, 1:]], axis=1)
        fulldata['jointlimit'] = np.concatenate([fulldata['jointlimit'], data['jointlimit'][:, 1:]], axis=1)
        # Join the slack trajectories - join as with the states
        fulldata['slacks'] = np.concatenate([fulldata['slacks'], data['slacks'][:, 1:]], axis=1)
    
    return fulldata

def a1_full_step_fast_optimization():
    savedir = os.path.join('examples','a1','foot_tracking_fast_shift','fullstep')
    a1 = opttools.make_a1()
    feet, _, qtraj = a1_fast_shifted_steps(a1)
    foot, step2 = feet[1], feet[2]
    for k, step in enumerate(step2):
        foot[k] = np.concatenate([foot[k], step[:, 1:]], axis=1)
    q = np.concatenate([qtraj[1], qtraj[2][:, 1:]], axis=1)
    # Get the previous trajectories as a warmstart
    sourcedir = os.path.join('examples','a1','foot_tracking_fast_shift')
    subdirs = ['firststep','secondstep']
    file = os.path.join('weight_1e+03','trajoptresults.pkl')
    dataset = [utils.load(utils.FindResource(os.path.join(sourcedir, subdir, file))) for subdir in subdirs]
    warmstart = join_backward_euler_trajectories(dataset)
    duration = warmstart['time'][-1]
    trajopt = setup_foot_tracking_gait(a1, foot, q[:6, :], duration, warmstart)
    # Add periodicity constraints
    Au, bu = periodicity_parameters(trajopt.u.shape[0])
    Al, bl = periodicity_parameters(trajopt.l.shape[0])
    Aj, bj = periodicity_parameters(trajopt.jl.shape[0])
    trajopt.prog.AddLinearEqualityConstraint(Au, bu, np.concatenate([trajopt.u[:, 0], trajopt.u[:, -1]], axis=0)).evaluator().set_description('control_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Al, bl, np.concatenate([trajopt.l[:, 0], trajopt.l[:, -1]], axis=0)).evaluator().set_description('force_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Aj, bj, np.concatenate([trajopt.jl[:, 0], trajopt.jl[:, -1]], axis=0)).evaluator().set_description('jointlimit_periodicity')
    # Solve the problem
    weights = [1, 1e1, 1e2, 1e3]
    opttools.progressive_solve(trajopt, weights, savedir)

def a1_multistep_optimization(numsteps = 10):
    savedir = os.path.join('examples','a1','foot_tracking_fast_shift',f'multistep_{numsteps}')
    a1 = opttools.make_a1()
    # Get the foot trajectories
    print('loading foot trajectories')
    feet, _, qtraj = a1_fast_shifted_steps(a1, numsteps)
    foot = join_foot_trajectories(feet[1:-1])
    q = join_configuration_trajectories(qtraj[1:-1])
    # Get the previous trajectories as a warmstart
    print('generating warmstart')
    sourcedir = os.path.join('examples','a1','foot_tracking_fast_shift','fullstep','weight_1e+03','trajoptresults.pkl')
    dataset = [utils.load(utils.FindResource(sourcedir))]
    data = copy.deepcopy(dataset)
    for _ in range(numsteps-1):
        dataset.extend(copy.deepcopy(data))
    warmstart = join_backward_euler_trajectories(dataset)
    duration = warmstart['time'][-1]
    print("Creating trajectory optimization")
    trajopt = setup_foot_tracking_gait(a1, foot, q[:6, :], duration, warmstart)
    # Add periodicity constraints
    print("Adding periodicity constraints")
    Au, bu = periodicity_parameters(trajopt.u.shape[0])
    Al, bl = periodicity_parameters(trajopt.l.shape[0])
    Aj, bj = periodicity_parameters(trajopt.jl.shape[0])
    trajopt.prog.AddLinearEqualityConstraint(Au, bu, np.concatenate([trajopt.u[:, 0], trajopt.u[:, -1]], axis=0)).evaluator().set_description('control_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Al, bl, np.concatenate([trajopt.l[:, 0], trajopt.l[:, -1]], axis=0)).evaluator().set_description('force_periodicity')
    trajopt.prog.AddLinearEqualityConstraint(Aj, bj, np.concatenate([trajopt.jl[:, 0], trajopt.jl[:, -1]], axis=0)).evaluator().set_description('jointlimit_periodicity')
    # Solve the problem
    weights = [1, 1e1, 1e2, 1e3]
    opttools.progressive_solve(trajopt, weights, savedir)

if __name__ == '__main__':
    #a1_first_step_fast_optimization()
    #a1_second_step_fast_optimization()
    #a1_full_step_fast_optimization()
    a1_multistep_optimization()