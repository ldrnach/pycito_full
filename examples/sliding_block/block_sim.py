"""
simulation examples using the sliding block as a base example for terrain estimation

Luke Drnach
November 9, 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import utilities as utils
from systems.timestepping import TimeSteppingMultibodyPlant
from systems.terrain import FlatTerrain, StepTerrain, VariableFrictionFlatTerrain

def terrain_sim(file, x0, dt, u, terrain):
    plant = TimeSteppingMultibodyPlant(file, terrain)
    plant.Finalize()
    t, x, f = plant.simulate(dt, x0, u, N = u.shape[1])
    return (t, x, f)

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

def compare_trajectories(t, x_sim, x_opt, f_sim, f_opt, title):
    _, axs = plt.subplots(4,1)
    labels = ["H-Pos", "V-Pos", "H-Vel", "V-Vel"]
    # Trajectory Figure
    for n in range(0, 4):
        axs[n].plot(t, x_sim[n,:], 'b-', linewidth=1.5, label="sim")
        axs[n].plot(t, x_opt[n,:], 'r-', linewidth=1.5, label="opt")
        axs[n].set_ylabel(labels[n])
    axs[-1].set_xlabel("Time (s)")
    axs[0].legend()
    axs[0].set_title(title)
    # Forces figure
    _, faxs = plt.subplots(3,1)
    D = np.array([[1, 0, 0, 0, 0],[0, 1, 0, -1, 0], [0, 0, 1, 0, -1]])
    f_sim = D.dot(f_sim)
    f_opt = D.dot(f_opt)
    flabels = ['Normal', 'Tangent_X','Tangent_Y']
    for n in range(0, 3):
        faxs[n].plot(t, f_sim[n,:], 'b-', linewidth=1.5, label="sim")
        faxs[n].plot(t, f_opt[n,:], 'r-', linewidth=1.5, label="opt")
        faxs[n].set_ylabel(flabels[n])
    faxs[-1].set_xlabel("Time (s)")
    faxs[0].legend()
    faxs[0].set_title(title)
    plt.show()

def traj_to_dict(t, x, f, u):
    return {"time": t,
            "state": x,
            "force": f,
            "control": u}

def calc_terrain_height(x, terrain):
    "Calculate terrain height across a trajectory"
    heights = np.zeros((x.shape[1],))
    for n in range(0,x.shape[1]):
        pt = terrain.nearest_point(x[0:3,n])
        heights[n] = pt[2]
    return heights

def calc_terrain_friction(x, terrain):
    "Calculate friction coefficient across a trajectory"
    fric = np.zeros((x.shape[1],))
    for n in range(0, x.shape[1]):
        fric[n] = terrain.get_friction(x[0:3,n])
    return fric

if __name__ == "__main__":
    # Setup
    file = "systems/urdf/sliding_block.urdf" 
    opt_traj = utils.load("data/slidingblock/block_trajopt.pkl")
    x0 = opt_traj["state"][:,0]
    u = opt_traj["control"]
    dt = 0.01
    terrains = [FlatTerrain(),
                VariableFrictionFlatTerrain(fric_func=low_friction),
                VariableFrictionFlatTerrain(fric_func=high_friction),
                StepTerrain(step_height=-0.5, step_location=2.5)]
    labels = ["FlatTerrain", "LowFriction", "HighFriction", "StepTerrain"]
    simdata = {}
    for terrain, label in zip(terrains, labels):
        # Run the simulations
        t, x, f = terrain_sim(file, x0, dt, u, terrain)
        # Plot the results
        compare_trajectories(t, x, opt_traj["state"], f, opt_traj["force"][0:5,:], label)
        # Convert the results to a dictionary and save
        simdata[label] = traj_to_dict(t, x, f, u)
        # Add in friction and terrain height to the results dictionary
        simdata[label]["height"] = calc_terrain_height(x, terrain)
        simdata[label]["friction"] = calc_terrain_friction(x, terrain)
    # Save the resulting trajectory optimizations
    utils.save("data/slidingblock/block_terrain_sims.pkl", simdata)