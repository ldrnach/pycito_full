
import os, copy
from tabnanny import check
import numpy as np

from pydrake.all import PiecewisePolynomial as pp
import pycito.trajopt.contactimplicit as ci

import pycito.utilities as utils
from pycito.systems.A1.a1 import A1VirtualBase

keys = ['state','control','force','slacks','jointlimit']

def concatenate_data(dataset):
    alldata = dataset.pop(0)
    for data in dataset:
        t_new = data['time'][1:] + alldata['time'][-1]
        alldata['time'] = np.concatenate([alldata['time'], t_new], axis=0)
        for key in keys:
            alldata[key] = np.concatenate([alldata[key], data[key][:, 1:]], axis=1)
    return alldata

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
    #viewer.plot_constraints(show_duals = False, savename = savename)


def main(numsteps=1):
    dir = os.path.join('examples','a1','foot_tracking_gait')
    subdirs = ['twostepopt']
    filepart = os.path.join('weight_1e+03', 'trajoptresults.pkl')
    data = [utils.load(utils.FindResource(os.path.join(dir, subdir, filepart))) for subdir in subdirs]
    copy_data = copy.deepcopy(data)
    for _ in range(numsteps-1):
        data.extend(copy.deepcopy(copy_data))
    data = join_backward_euler_trajectories(data)
    trajdata = {}
    trajdata['state'] = pp.FirstOrderHold(data['time'], data['state'])
    for key in keys[1:]:
        trajdata[key] = pp.ZeroOrderHold(data['time'], data[key])
    # Plot the data
    a1 = A1VirtualBase()
    a1.Finalize()
    savedir = os.path.join(dir, f'{2*numsteps}step_plots')
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    a1.plot_trajectories(trajdata['state'], trajdata['control'], trajdata['force'], trajdata['jointlimit'], show=False, savename=os.path.join(savedir, 'vis.png'))
    utils.save(os.path.join(savedir, 'combinedresults.pkl'), data)
    a1.visualize(trajdata['state'])
    # Plot the constraints
    check_constraint_satisfaction(data, os.path.join(savedir, 'constraints.png'))

if __name__ == "__main__":
    main(numsteps=5)
