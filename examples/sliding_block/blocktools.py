"""
General tools for running examples with the sliding block

Luke Drnach
April 11, 2022
"""
import os
from pydrake.all import PiecewisePolynomial as pp

from pycito.systems.block.block import Block
import pycito.systems.terrain as terrain
import pycito.systems.contactmodel as cm
import pycito.utilities as utils

SIMNAME = 'simdata.pkl'

def low_friction(x):
    if x[0] < 2.0 or x[0] > 4.0:
        return 0.5
    else:
        return 0.1

def high_friction(x):
    if x[0] < 2.0 or x[0] > 4.0:
        return 0.5
    else:
        return 0.9

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

def save_sim_data(tsim, xsim, usim, fsim, status, savedir=None):
    simdata = {'time': tsim,
                'state': xsim,
                'control': usim,
                'force': fsim,
                'status': status}
    if savedir is not None:
        filename = os.path.join(savedir, SIMNAME)
        utils.save(filename, simdata)
        print(f"Simulation data saved at {filename}")

def make_flatterrain_model():
    block = Block()
    block.Finalize()
    return block

def make_lowfriction_model():
    lowfric = terrain.VariableFrictionFlatTerrain(height = 0, fric_func = low_friction)
    block = Block(terrain = lowfric)
    block.Finalize()
    return block

def make_highfriction_model():
    highfric = terrain.VariableFrictionFlatTerrain(height = 0, fric_func = high_friction)
    block = Block(terrain = highfric)
    block.Finalize()
    return block

def make_stepterrain_model():
    stepterrain = terrain.StepTerrain(step_height = -0.5, step_location=2.5)
    block = Block(terrain = stepterrain)
    block.Finalize()
    return block

def make_semiparametric_block_model():
    block = Block()
    block.Finalize()
    block.terrain = cm.SemiparametricContactModel.FlatSurfaceWithRBFKernel(friction = 0.5, length_scale = 0.1, reg=0.01)
    return block 

if __name__ == '__main__':
    print("Hello from blocktools")