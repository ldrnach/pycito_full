"""
Generate a visualization of the sliding block trajectory optimization results

Luke Drnach
February 9, 2022
"""

import os

from pydrake.all import PiecewisePolynomial as pp

from pycito.systems.block.block import Block
import pycito.utilities as utils


def visualize_block_results():
    # Get the trajectory data
    sourcedir = os.path.join('data','slidingblock','block_trajopt.pkl')
    data = utils.load(utils.FindResource(sourcedir))
    xtraj = pp.FirstOrderHold(data['time'], data['state'])
    utraj = pp.ZeroOrderHold(data['time'], data['control'])
    ftraj = pp.ZeroOrderHold(data['time'], data['force'])
    # Visualize
    targetdir = os.path.join('examples','sliding_block','optimization')
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    block = Block()
    block.Finalize()
    block.plot_trajectories(xtraj, utraj, ftraj, show=False, savename=os.path.join(targetdir, 'opt.png'))
    block.visualize(xtraj)
    utils.save(os.path.join(targetdir, 'trajoptresult.pkl'), data)

if __name__ == '__main__':
    visualize_block_results()

