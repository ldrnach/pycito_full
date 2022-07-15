import os, copy
import numpy as np
from math import ceil
from pycito.utilities import load
from pycito.controller.mpc import LinearizedContactTrajectory
from pycito.systems.A1.a1 import A1VirtualBase
import pycito.trajopt.contactimplicit as ci

DISTANCE = 3
SOURCE = os.path.join('data','a1','ellipse_foot_tracking','fast','fullstep','weight_1e+03','trajoptresults.pkl')
TARGETDIR = os.path.join('data','a1','reference','fast',f'{DISTANCE}m')
TARGETNAME = 'reftraj.pkl'

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

def check_constraint_satisfaction(data, savename):
    A1 = A1VirtualBase()
    A1.Finalize()
    dt = np.diff(data['time'])
    options = ci.OptimizationOptions()
    options.useLinearComplementarityWithCost()
    trajopt = ci.ContactImplicitDirectTranscription(A1, 
                                                    A1.multibody.CreateDefaultContext(), 
                                                    num_time_samples = dt.size+1,
                                                    minimum_timestep = np.min(dt),
                                                    maximum_timestep = np.max(dt),
                                                    options=options)
    viewer = ci.ContactConstraintViewer(trajopt, data)
    cstr = viewer.calc_constraint_values()
    viewer.plot_dynamic_defects(data['time'], cstr['dynamics'], show=False, savename=savename)

def calc_num_steps(data):
    dx = data['state'][0,-1] - data['state'][0,0]
    nstep = ceil(DISTANCE/dx)
    print(f"Moving {dx:0.2f}m per step, it will take {nstep} steps to achieve {DISTANCE}m")
    return nstep

def main():
    if not os.path.exists(TARGETDIR):
        os.makedirs(TARGETDIR)
    data = load(SOURCE)
    nsteps = calc_num_steps(data)
    # Create the full dataset
    data = [data]
    cdata = copy.deepcopy(data)
    print(f"Repeating data")
    for _ in range(nsteps-1):
        data.extend(copy.deepcopy(cdata))
    print(f"Joining steps")
    data = join_backward_euler_trajectories(data)
    # Check that the constraints are all satisfied
    check_constraint_satisfaction(data, savename=os.path.join(TARGETDIR,'constraints.png'))
    # Create the reference trajectory
    a1 = A1VirtualBase()
    a1.terrain.friction = 1.0
    a1.Finalize()
    print(f"Linearizing the reference trajectory")
    lintraj = LinearizedContactTrajectory(a1, data['time'],data['state'], data['control'],data['force'],data['jointlimit'])
    lintraj.save(os.path.join(TARGETDIR, TARGETNAME))
    print(f"Saved linearized reference trajectory to {os.path.join(TARGETDIR, TARGETNAME)}")

if __name__ == '__main__':
    main()