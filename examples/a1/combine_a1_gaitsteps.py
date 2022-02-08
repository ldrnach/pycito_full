
import os
import numpy as np

from pydrake.all import PiecewisePolynomial as pp

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

def main(numsteps=1):
    dir = os.path.join('examples','a1','foot_tracking_gait')
    subdirs = ['first_step','second_step_continuous']
    filepart = os.path.join('weight_1e+03', 'trajoptresults.pkl')
    data = [utils.load(utils.FindResource(os.path.join(dir, subdir, filepart))) for subdir in subdirs]
    data = data * numsteps
    data = concatenate_data(data)
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


if __name__ == "__main__":
    main(numsteps=2)
