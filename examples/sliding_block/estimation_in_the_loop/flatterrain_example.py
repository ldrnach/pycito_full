"""
Luke Drnach
April 13, 2022
"""

import os
import numpy as np
from pycito.systems.block.block import Block
import pycito.utilities as utils
import estimation_control_tools as campctools
import campc_animation_tools as animator
from pycito.controller.optimization import OptimizationLogger
import pycito.systems.kernels as kernels

SIM_DURATION = 1.5
TARGET = os.path.join('examples','sliding_block','estimation_in_the_loop','paper','flatterrain')
ANIMATION_NAME = 'campc_animation.mp4'
MPCANIMATIONNAME = 'mpc_animation.mp4'

def make_flatterrain_model():
    block = Block()
    block.Finalize()
    return block

def main():
    W = np.diag([0.1, 0.1, 0.0])
    surfkernel = kernels.CompositeKernel(
        kernels.CenteredLinearKernel(W),
        kernels.ConstantKernel(1),
        kernels.WhiteNoiseKernel(0.01)
    )
    frickernel = kernels.RegularizedConstantKernel(1, 0.01)
    globalkernel = kernels.RegularizedPseudoHuberKernel(length_scale=np.array([0.1, 0.1, np.inf]), delta=0.1, noise=0.01)
    sp_contact = campctools.make_semiparametric_contact_model(surfkernel, frickernel)
    campctools.run_piecewise_estimation_control(make_flatterrain_model(), 
                                    sp_contact,
                                    globalkernel,
                                    savedir = TARGET)

def main_ambiguity():
    plant = make_flatterrain_model()
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
    truemodel = make_flatterrain_model()
    # Load the appropriate reference trajectories
    reftraj = campctools.make_mpc_controller().lintraj
    esttraj = campctools.make_estimator_controller().getContactEstimationTrajectory()
    esttraj.loadEstimatedTrajectory(os.path.join(TARGET, campctools.TRAJNAME))
    # Setup the animator
    animation = animator.BlockCAMPCComparisonAnimator(truemodel, reftraj, esttraj)
    animation.animate(mpclogs, campc_full, savename=os.path.join(TARGET, ANIMATION_NAME))
    mpc_animation = animator.BlockMPCAnimator(truemodel, reftraj)
    mpc_animation.animate(mpclogs, savename=os.path.join(TARGET, MPCANIMATIONNAME))

if __name__ == '__main__':
    main()