
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycito.systems.block.block import Block
import pycito.systems.terrain as terrain
import examples.sliding_block.estimation_in_the_loop.estimation_control_tools as campctools
import pycito.systems.kernels as kernels
from pycito.utilities import load
from pycito.controller.optimization import OptimizationLogger
from pycito.controller.contactestimator import EstimatedContactModelRectifier

SOURCE = os.path.join("examples","sliding_block","estimation_in_the_loop","stepterrain",'centeredaffinekernel_tuned')
REFDATA = 'campcsim.pkl'
ESTRAJ = 'estimatedtrajectory.pkl'
LOGDATA = os.path.join('campc_logs','EstimationLogs.pkl')
INDEX = 80
GLOBAL_MODEL = False

def make_stepterrain_model():
    stepterrain = terrain.StepTerrain(step_height = -0.5, step_location=2.5)
    block = Block(terrain = stepterrain)
    block.Finalize()
    return block

# Make the true terrain model
truemodel = make_stepterrain_model()

# Make the semiparametric model
W = np.diag([0.1, 0.1, 0.0])
surfkernel = kernels.CompositeKernel(kernels.CenteredLinearKernel(W), kernels.ConstantKernel(1), kernels.WhiteNoiseKernel(0.01))
frickernel = kernels.RegularizedConstantKernel(1.0, 0.01)

sp_contact = campctools.make_semiparametric_contact_model(surfkernel, frickernel)

# Load the contact estimation trajectory and get the final model
estraj = campctools.load_estimation_trajectory(SOURCE)
logs = OptimizationLogger.load(os.path.join(SOURCE, LOGDATA)).logs
cpts = np.concatenate(estraj.get_contacts(INDEX - 5, INDEX), axis=1)
dweights = logs[INDEX]['distance_weights']
fweights = logs[INDEX]['friction_weights']

sp_contact.add_samples(cpts, dweights, fweights)


# Plot the results
terrain_start = np.array([0, 0, 0])
terrain_end = np.array([6, 0, 0])
terrain_samples = np.linspace(terrain_start, terrain_end, 701).T
true_terrain = truemodel.terrain.find_surface_zaxis_zeros(terrain_samples)
sp_terrain = sp_contact.find_surface_zaxis_zeros(terrain_samples)

fig, axs = plt.subplots(2,1)
axs[0].plot(true_terrain[0,:], true_terrain[2,:], 'g-', linewidth=1.5, label='Actual')
axs[0].plot(sp_terrain[0, :], sp_terrain[2,:], 'b-', linewidth=1.5, label='SP_Linear')

axs[0].set_ylabel('Terrain (m)')
axs[0].set_xlabel('Position (m)')

axs[0].legend(frameon=False)
plt.show()