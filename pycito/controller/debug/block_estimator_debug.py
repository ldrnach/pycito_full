import os
import numpy as np
from collections import Counter

from pycito import utilities as utils
import pycito.controller.contactestimator as ce
from pycito.controller.speedtesttools import SpeedTestResult 
from pycito.systems.block.block import Block
from pycito.systems.contactmodel import SemiparametricContactModel

TESTDATASOURCE = os.path.join('examples','sliding_block','estimator_speedtests')
TESTDATANAME = 'speedtestresults.pkl'
SIMSOURCE = os.path.join('examples','sliding_block','simulations')
SIMNAME = os.path.join('openloop','simdata.pkl')

def create_reference_trajectory(sourcepart):
    # Load and truncate the data
    data = utils.load(os.path.join(SIMSOURCE, sourcepart, SIMNAME))
    data['time'] = data['time'][:111]
    for key in ['state', 'control', 'force']:
        data[key] = data[key][:111]
    # Convert to estimation trajectory
    block = Block()
    block.Finalize()
    block.terrain = SemiparametricContactModel.FlatSurfaceWithRBFKernel()
    traj = ce.ContactEstimationTrajectory(block, data['state'][:, 0])
    for t, x, u in zip(data['time'][1:], data['state'][:, 1:].T, data['control'][:, 1:].T):
        traj.append_sample(t, x, u)
    return traj

def rerun_test(traj, sample_number, horizon_number):
    nstarts = 30
    N = traj.num_timesteps
    start_ptr = N - nstarts
    subtraj = traj.subset(0, start_ptr + sample_number)
    estimator = ce.ContactModelEstimator(subtraj, horizon_number)
    estimator.create_estimator()
    print(f"Re-solving contact estimation")
    result = estimator.solve()
    return estimator, result

def print_result(estimator, result):
    utils.printProgramReport(result, estimator._prog, terminal=True, verbose=True)
    # Print the solution
    alpha = result.GetSolution(estimator._distance_weights)
    beta = result.GetSolution(estimator._friction_weights)
    force = result.GetSolution(estimator.forces)
    vel = result.GetSolution(estimator.velocities)
    feas = result.GetSolution(estimator.feasibilities)
    print(f"Distance weights = {alpha}")
    print(f"Friction weights = {beta}")
    print(f"Reaction forces = {force}")
    print(f"Tangential velocities = {vel}")
    print(f"Feasibility = {feas}")
    # Final terrain errors
    print(f"Distance errors = {estimator._distance_kernel.dot(alpha)}")
    print(f"Friction errors = {estimator._friction_kernel.dot(beta)}")
    # Find and calculate the semiparametric friction cone error
    for cstr in estimator._prog.GetAllConstraints():
        if cstr.evaluator().get_description() == 'SemiparametricFrictionCone':
            dvars = cstr.variables()
            dvals = result.GetSolution(dvars)
            out = cstr.evaluator().Eval(dvals)
            print(f"SemiparametricFrictionConeConstraint = {out}")

def check_infeasible_runs(traj, starts, horizon):
    all_infeas = []
    for start in starts:
        estimator, result = rerun_test(traj, start, horizon)
        infeasibles = result.GetInfeasibleConstraintNames(estimator._prog)
        infeas = [name.split("[")[0] for name in infeasibles]
        all_infeas.extend(list(set(infeas)))        
    # Count all the infeasible constraints
    counts = Counter(all_infeas)
    print(f"Of the {starts.size} failed problems")
    for key, value in counts.items():
        print(f"\t{value} failed to satisfy the {key} constraint")

def debug_main():
    sourcepart = 'flatterrain'
    # Load the reference trajectory data
    reftraj = create_reference_trajectory(sourcepart)
    # Load the speed test data
    source = os.path.join(TESTDATASOURCE, sourcepart, TESTDATANAME)
    result = SpeedTestResult.load(source)
    horizon = 3
    infos = result.solve_info[:, horizon-1].astype(int)
    success = np.where(infos == 1)[0]
    fails = np.where(infos == 13)[0]
    print(f"Successful solves = {success}")
    print(f"Failed solves = {fails}")
    # Check the failure case
    estimator, new_result = rerun_test(reftraj, fails[0], horizon)
    print_result(estimator, new_result)
    # Check the successful case
    estimator, new_result = rerun_test(reftraj, success[0], horizon)
    print_result(estimator, new_result)
    check_infeasible_runs(reftraj, fails, horizon)


if __name__ == '__main__':
    debug_main()