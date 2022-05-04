"""
Luke Drnach
April 13, 2022
"""

import os
import numpy as np
from pycito.systems.block.block import Block
import estimation_control_tools as campctools
import pycito.systems.terrain as terrain
import pycito.utilities as utils
import campc_animation_tools as animator
from pycito.controller.optimization import OptimizationLogger
import pycito.systems.kernels as kernels 

SIM_DURATION = 1.5
TARGET = os.path.join('examples','sliding_block','estimation_in_the_loop','low_friction','rbfkernel')
ANIMATION_NAME = 'campc_animation.mp4'
MPCANIMATIONNAME = 'mpc_animation.mp4'

def low_friction(x):
    if x[0] < 2.0 or x[0] > 4.0:
        return 0.5
    else:
        return 0.1

def make_lowfriction_model():
    lowfric = terrain.VariableFrictionFlatTerrain(height = 0, fric_func = low_friction)
    block = Block(terrain = lowfric)
    block.Finalize()
    return block

def main():
    campctools.run_estimation_control(make_lowfriction_model(), 
                                    kernel = kernels.RegularizedRBFKernel(length_scale=np.array([0.1, 0.1, np.inf]), noise = 0.01),
                                    savedir = TARGET)


def main_ambiguity():
    plant = make_lowfriction_model()
    traj = campctools.load_estimation_trajectory(TARGET)
    model = campctools.run_ambiguity_optimization(traj)
    data = utils.load(os.path.join(TARGET, 'campcsim.pkl'))
    pts = campctools.get_x_samples(data, sampling=1000)
    campctools.compare_estimated_contact_model(model, plant.terrain, pts, savedir=TARGET, name='contactmodelwithambiguity')
    # Save the contact model
    utils.save(os.path.join(TARGET, 'contactambiguity.pkl'), model)

def main_animation():
    # Load the log files
    mpcfile = os.path.join(TARGET, 'mpc_logs', campctools.CONTROLLOGNAME)
    campcfile = os.path.join(TARGET, 'campc_logs', campctools.CONTROLLOGNAME)
    estfile = os.path.join(TARGET, 'campc_logs', campctools.ESTIMATELOGNAME)
    mpclogs = OptimizationLogger.load(mpcfile).logs
    campclogs = OptimizationLogger.load(campcfile).logs
    estlogs = OptimizationLogger.load(estfile).logs
    campc_full = [{**camplog, **estlog} for camplog, estlog in zip(campclogs, estlogs)]
    # Get the block model
    truemodel = make_lowfriction_model()
    # Load the appropriate reference trajectories
    reftraj = campctools.make_mpc_controller().lintraj
    esttraj = campctools.make_estimator_controller().getContactEstimationTrajectory()
    esttraj.loadEstimatedTrajectory(os.path.join(TARGET, campctools.TRAJNAME))
    # Setup the animator
    animation = animator.BlockCAMPCComparisonAnimator(truemodel, reftraj, esttraj)
    animation.animate(mpclogs, campc_full, savename=os.path.join(TARGET, ANIMATION_NAME))
    # Setup MPC only animation
    # mpc_animation = animator.BlockMPCAnimator(truemodel, reftraj)
    # mpc_animation.animate(mpclogs, savename=os.path.join(TARGET, MPCANIMATIONNAME))


if __name__ == '__main__':
    main()
    #main_ambiguity()
    # main_animation()