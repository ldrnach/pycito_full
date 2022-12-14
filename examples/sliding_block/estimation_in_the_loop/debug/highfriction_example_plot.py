import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import examples.sliding_block.estimation_in_the_loop.estimation_control_tools as campctools
from pycito.utilities import load
from pycito.controller.optimization import OptimizationLogger
from pycito.controller.contactestimator import EstimatedContactModelRectifier
from pycito.systems.block.block import Block
import pycito.systems.terrain as terrain

SOURCE = os.path.join("examples","sliding_block","estimation_in_the_loop","high_friction",'phkernel_global_tuned')
REFDATA = 'campcsim.pkl'
ESTRAJ = 'estimatedtrajectory.pkl'
LOGDATA = os.path.join('campc_logs','EstimationLogs.pkl')
INDEX = 60
GLOBAL_MODEL = True

def high_friction(x):
    if x[0] < 2.0 or x[0] > 4.0:
        return 0.5
    else:
        return 0.9

def make_highfriction_model():
    highfric = terrain.VariableFrictionFlatTerrain(height = 0, fric_func = high_friction)
    block = Block(terrain = highfric)
    block.Finalize()
    return block

truemodel = make_highfriction_model()
refmodel = campctools.make_block_model()
reftraj = campctools.make_mpc_controller().lintraj
# Plot the terrain
terrain_start = np.array([0, 0, 0])
terrain_end = np.array([6, 0, 0])
terrain_samples = np.linspace(terrain_start, terrain_end, 701).T
true_terrain = truemodel.terrain.find_surface_zaxis_zeros(terrain_samples)
ref_terrain = refmodel.terrain.find_surface_zaxis_zeros(terrain_samples)

fig, axs = plt.subplots(2,1)
axs[0].plot(true_terrain[0,:], true_terrain[2,:], 'g-', linewidth=1.5, label='Actual')
axs[0].plot(ref_terrain[0,:], ref_terrain[2,:], 'k--', linewidth=1.5, label='Reference')

# Plot the box state
states = reftraj._state
axs[0].plot(states[0,:], states[1,:], linewidth=1.5, color='grey', linestyle=':', label=None)
refbox = states[:2, INDEX]
axs[0].add_patch(Rectangle((refbox[0] - 0.5, refbox[1] - 0.5), 1, 1, edgecolor='black', facecolor='grey'))

# Load the simulation data and plot the box
campcsim = load(os.path.join(SOURCE, REFDATA))
axs[0].plot(campcsim['state'][0,:], campcsim['state'][1,:], linewidth=1.5, color='green',linestyle='-', label=None)
simbox = campcsim['state'][:2,INDEX]
axs[0].add_patch(Rectangle((simbox[0] - 0.5, simbox[1] - 0.5), 1, 1, edgecolor='black', facecolor='green'))
# Plot the estimated contact model height
estraj = campctools.load_estimation_trajectory(SOURCE)
if GLOBAL_MODEL:
    subtraj = estraj.subset(0, INDEX)
    rectifier = EstimatedContactModelRectifier(subtraj, surf_max = 10, fric_max = 2)
    sp_contact = rectifier.get_global_model()
else:
    logs = OptimizationLogger.load(os.path.join(SOURCE, LOGDATA)).logs
    cpts = np.concatenate(estraj.get_contacts(INDEX - 5, INDEX), axis=1)
    dweight = logs[INDEX]['distance_weights']
    fweight = logs[INDEX]['friction_weights']
    sp_contact = estraj.contact_model
    sp_contact.add_samples(cpts, dweight, fweight)

sp_terrain = sp_contact.find_surface_zaxis_zeros(terrain_samples)
axs[0].plot(sp_terrain[0,:], sp_terrain[2,:], linewidth=1.5, color='blue')

# Plot the friction coefficients
fric = np.concatenate([truemodel.terrain.eval_friction(pt) for pt in terrain_samples.T], axis=0)
sp_fric = np.concatenate([sp_contact.eval_friction(pt) for pt in terrain_samples.T], axis=0)

axs[1].plot(terrain_samples[0,:], fric, 'g-', linewidth=1.5)
axs[1].plot(terrain_samples[0,:], sp_fric, linewidth=1.5, color='blue')
axs[1].set_ylabel('Friction Coefficient')
axs[1].set_xlabel('Position')

axs[0].legend(frameon=False)
axs[0].set_aspect('equal')
plt.show()