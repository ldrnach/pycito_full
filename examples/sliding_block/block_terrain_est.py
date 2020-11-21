"""
Terrain estimation with the sliding block data

Luke Drnach
November 17, 2020
"""

from systems.terrainestimator import ResidualTerrainEstimator
from systems.terrain import GaussianProcessTerrain
from systems.timestepping import TimeSteppingMultibodyPlant
import systems.gaussianprocess as gp
import utilities as utils
import numpy as np
import matplotlib.pyplot as plt 

def get_data():
    """Loads the data from the pickle file"""
    data = utils.load("data/slidingblock/block_terrain_sims.pkl")
    return data

def make_block_gp_model():
    """Load the block URDF and make a timestepping model with a GP terrain"""
    # Specify the file
    file = "systems/urdf/sliding_block.urdf"
    # Setup the GP terrain
    height_gp = gp.GaussianProcess(xdim=2, 
                                mean=gp.ConstantFunc(0.),
                                kernel=gp.SquaredExpKernel(M=np.eye(2), s=1.))
    fric_gp = gp.GaussianProcess(xdim=2,
                                mean=gp.ConstantFunc(0.5),
                                kernel=gp.SquaredExpKernel(M=10*np.eye(2), s=0.1))
    terrain = GaussianProcessTerrain(height_gp, fric_gp)
    # Create the time-stepping model
    plant = TimeSteppingMultibodyPlant(file, terrain)
    plant.Finalize()
    return plant

def run_terrain_estimation(plant, t, x, u):
    """ runs residual terrain estimation and updates the plant"""
    estimator = ResidualTerrainEstimator(plant)
    for n in range(0, x.shape[1]-1):
        h = t[n+1]-t[n]
        estimator.estimate_terrain(h, x[:,n], x[:,n+1], u[:,n])
        
def plot_terrain_results(plant, x, title):
    """plot the results from terrain estimation"""
    # The coordinate axis
    x_1 = x[0:2,:]
    # The priors
    mp_h, Sp_h = plant.terrain.height.prior(x_1)
    mp_c, Sp_c = plant.terrain.friction.prior(x_1)
    # The posteriors
    mu_h, S_h = plant.terrain.height.posterior(x_1)
    mu_c, S_c = plant.terrain.friction.posterior(x_1)
    # The figure
    _, axs = plt.subplots(2,1)
    # Plot the priors
    gp.plot_gp(axs[0], x_1[0,:], mp_h, Sp_h)
    gp.plot_gp(axs[1], x_1[0,:], mp_c, Sp_c)
    axs[0].set_title("Prior")
    axs[0].set_ylabel("Terrain Height")
    axs[1].set_ylabel("Friction")
    axs[1].set_xlabel("Position")
    # Plot the posteriors
    _, axs2 = plt.subplots(2,1)
    gp.plot_gp(axs2[0], x_1[0,:], mu_h, S_h)
    gp.plot_gp(axs2[1], x_1[0,:], mu_c, S_c)
    # Add sample data to the posteriors
    axs2[0].plot(plant.terrain.height.data_x[0,:], plant.terrain.height.data_y[0,:],'x')
    axs2[1].plot(plant.terrain.friction.data_x[0,:], plant.terrain.friction.data_y[0,:],'x')
    # Labels
    axs2[0].set_title("Posterior" + " " + title)
    axs2[0].set_ylabel("Terrain Height")
    axs2[1].set_ylabel("Friction")
    axs2[1].set_xlabel("Position")
    # Show the figures
    plt.show()

if __name__ == "__main__":
    """Sliding block terrain estimation main function"""
    # Get the sample data
    data = get_data()
    for key in data.keys():
        # Make a new terrain model
        plant = make_block_gp_model()
        # unpack the data
        t = data[key]["time"]
        x = data[key]["state"]
        u = data[key]["control"]
        # Run terrain estimation
        run_terrain_estimation(plant, t, x, u)
        # Plot the results
        plot_terrain_results(plant, x, key)