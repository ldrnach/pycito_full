
import os, copy, time
import numpy as np
import matplotlib.pyplot as plt
from pycito.systems.block.block import Block
import pycito.systems.terrain as terrain
import examples.sliding_block.estimation_in_the_loop.estimation_control_tools as campctools
import pycito.systems.kernels as kernels
from pycito.utilities import load
from pycito.controller.optimization import OptimizationLogger
from pycito.controller.contactestimator import EstimatedContactModelRectifier
import pycito.systems.contactmodel as cm

SOURCE = os.path.join("examples","sliding_block","estimation_in_the_loop","stepterrain",'centeredaffinekernel_tuned')
REFDATA = 'campcsim.pkl'
ESTRAJ = 'estimatedtrajectory.pkl'
LOGDATA = os.path.join('campc_logs','EstimationLogs.pkl')
INDEX = 140
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

# Create a new, nested semiparametric model
ns_contact = cm.SemiparametricContactModel(
    surface = cm.SemiparametricModel(sp_contact.surface, kernels.RegularizedPseudoHuberKernel(length_scale = np.array([0.1, 0.1, np.inf]), delta = 0.1, noise = 0.01)),
    friction = cm.SemiparametricModel(sp_contact.friction, kernels.RegularizedPseudoHuberKernel(length_scale = np.array([0.1, 0.1, np.inf]), delta = 0.1, noise= 0.01))
)
# Update the contact estimation trajectory
# Store the existing model, distance, and friction coefficients
print(f"Updating contact model\t", end="", flush=True)
start = time.perf_counter()
subtraj = estraj.subset(0, INDEX)
for k, cpt in enumerate(subtraj._contactpoints):
    dk = sp_contact.surface_kernel(cpt, cpts).dot(dweights)
    df = sp_contact.friction_kernel(cpt, cpts).dot(fweights)
    subtraj._distance_error[k] -= dk
    subtraj._friction_error[k] -= df
    subtraj._distance_cstr[k] += dk
    subtraj._friction_cstr[k] += df

subtraj.contact_model = ns_contact
print(f"Elapsed: {time.perf_counter() - start:0.2e}", end="\n", flush=True)
print(f"Solving global contact estimation\t", end="", flush=True)
start = time.perf_counter()
rectifier = EstimatedContactModelRectifier(subtraj, surf_max = 2, fric_max = 2)
rectifier.useSnoptSolver()
rectifier.setSolverOptions({'Major Feasibility Tolerance': 1e-4, 
                            'Major Optimality Tolerance': 1e-4})
pw_model = rectifier.solve_global_model_with_ambiguity()
lb_model = pw_model.lower_bound
dweights = pw_model.surface._kernel_weights
fweights = pw_model.friction._kernel_weights
print(f"Elapsed: {time.perf_counter() - start:0.2e}", end="\n", flush=True)

# Plot the results
terrain_start = np.array([0, 0, 0])
terrain_end = np.array([6, 0, 0])
terrain_samples = np.linspace(terrain_start, terrain_end, 701).T
print(f"Evaluating true terrain\t", end="", flush=True)
start = time.perf_counter()
true_terrain = truemodel.terrain.find_surface_zaxis_zeros(terrain_samples)
print(f"Elapsed: {time.perf_counter() - start:0.2e}", end="\n", flush=True)
print(f"Evaluating linear terrain\t", end="", flush=True)
start = time.perf_counter()
sp_terrain = sp_contact.find_surface_zaxis_zeros(terrain_samples)
print(f"Elapsed: {time.perf_counter() - start:0.2e}", end="\n", flush=True)
print(f"Evaluating piecewise terrain\t", end="", flush=True)
start = time.perf_counter()
pw_terrain = pw_model.find_surface_zaxis_zeros(terrain_samples)
print(f"Elapsed: {time.perf_counter() - start:0.2e}", end="\n", flush=True)
print(f"Evaluating upper_bound terrain\t", end="", flush=True)
start = time.perf_counter()
ub_terrain = lb_model.find_surface_zaxis_zeros(terrain_samples)
cpts_all = np.concatenate(estraj.get_contacts(0, INDEX), axis=1)
print(f"Elapsed: {time.perf_counter() - start:0.2e}")
print(f"Elapsed: {time.perf_counter() - start:0.2e}", end="\n", flush=True)
print(f"Making plots")
fig, axs = plt.subplots(2,1)
axs[0].plot(true_terrain[0,:], true_terrain[2,:], 'g-', linewidth=1.5, label='Actual')
axs[0].plot(cpts_all[0,:], cpts_all[2,:], '*', label='Contacts')
axs[0].plot(sp_terrain[0, :], sp_terrain[2,:], 'b-', linewidth=1.5, label='SP_Linear')
axs[0].plot(pw_terrain[0, :], pw_terrain[2,:], 'r-', linewidth=1.5, label='Piecewise')
axs[0].plot(ub_terrain[0,:], ub_terrain[2,:], 'y-', linewidth=1.5, label='UpperBound')

axs[0].set_ylabel('Terrain (m)')
axs[0].set_xlabel('Position (m)')

pw_model.compress(atol=1e-3)
axs[1].plot(true_terrain[0,:], true_terrain[2,:], linewidth=1.5, label='Actual')
axs[1].plot(pw_terrain[0,:], pw_terrain[2,:], linewidth=1.5, label='Full Model')
print(f"Evaluating compressed terrain\t", end="", flush=True)
start = time.perf_counter()
cw_terrain = pw_model.find_surface_zaxis_zeros(terrain_samples)
print(f"Elapsed: {time.perf_counter() - start:0.2e}", end="\n", flush=True)
axs[1].plot(cw_terrain[0,:], cw_terrain[2,:], linewidth=1.5, label='Compressed Model')

print(f"Compressed model has {pw_model.surface._kernel_weights.shape[0]}/{INDEX} surface samples and {pw_model.friction._kernel_weights.shape[0]}/{INDEX} friction samples ")

axs[0].legend(frameon=False)
axs[1].legend(frameon=False)
plt.show()

fig, axs = plt.subplots(2,1)
axs[0].hist(dweights, bins=20)
axs[0].set_xlabel("Distance Weights")
axs[0].set_ylabel('Frequency')
axs[1].hist(fweights, bins=20)
axs[1].set_xlabel('Friction weights')
axs[1].set_ylabel('Frequency')
plt.show()
