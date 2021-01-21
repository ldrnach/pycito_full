"""
Terrain estimation with the sliding block data

Luke Drnach
November 17, 2020
"""
import timeit
from systems.terrainestimator import ResidualTerrainEstimator
from systems.terrain import GaussianProcessTerrain
from systems.block.block import Block
import systems.gaussianprocess as gp
import utilities as utils
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def get_data():
    """Loads the data from the pickle file"""
    data = utils.load("data/slidingblock/block_terrain_sims.pkl")
    return data

def make_block_gp_model():
    """Load the block URDF and make a timestepping model with a GP terrain"""
    # Setup the GP terrain
    height_gp, fric_gp = make_gp_models()
    # Create the time-stepping model
    plant = Block(terrain=GaussianProcessTerrain(height_gp, fric_gp))
    plant.Finalize()
    return plant

def make_gp_models():
    height_gp = gp.GaussianProcess(xdim=2, 
                                mean=gp.ConstantFunc(0.),
                                kernel=gp.SquaredExpKernel(M=np.eye(2), s=1.))
    fric_gp = gp.GaussianProcess(xdim=2,
                                mean=gp.ConstantFunc(0.5),
                                kernel=gp.SquaredExpKernel(M=10*np.eye(2), s=0.1))
    return (height_gp, fric_gp)

def run_terrain_estimation(plant, t, x, u):
    """ runs residual terrain estimation and updates the plant"""
    estimator = ResidualTerrainEstimator(plant)
    for n in range(0, x.shape[1]-1):
        h = t[n+1]-t[n]
        estimator.estimate_terrain(h, x[:,n], x[:,n+1], u[:,n])
        
def plot_terrain_results(plant, data, key):
    """plot the results from terrain estimation"""
    x = data[key]["state"]
    h = data[key]["height"]
    m = data[key]["friction"]
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
    gp.plot_gp(axs[0], x_1[0,:], mp_h, Sp_h,label="GP")
    gp.plot_gp(axs[1], x_1[0,:], mp_c, Sp_c,label="GP")
    axs[0].plot(x_1[0,:], h, 'g--', linewidth=1.5,label="True")
    axs[1].plot(x_1[0,:], m, 'g--', linewidth=1.5,label="True")
    axs[0].set_title("Prior")
    axs[0].set_ylabel("Terrain Height")
    axs[1].set_ylabel("Friction")
    axs[1].set_xlabel("Position")
    axs[0].legend()
    # Plot the posteriors
    _, axs2 = plt.subplots(2,1)
    gp.plot_gp(axs2[0], x_1[0,:], mu_h, S_h, label="GP")
    gp.plot_gp(axs2[1], x_1[0,:], mu_c, S_c, label="GP")
    axs2[0].plot(x_1[0,:], h, 'g--', linewidth=1.5,label="True")
    axs2[1].plot(x_1[0,:], m, 'g--', linewidth=1.5,label="True")
    # Add sample data to the posteriors
    axs2[0].plot(plant.terrain.height.data_x[0,:], plant.terrain.height.data_y[0,:],'x')
    axs2[1].plot(plant.terrain.friction.data_x[0,:], plant.terrain.friction.data_y[0,:],'x')
    # Labels
    axs2[0].set_title("Posterior" + " " + key)
    axs2[0].set_ylabel("Terrain Height")
    axs2[1].set_ylabel("Friction")
    axs2[1].set_xlabel("Position")
    axs2[0].legend()
    # Show the figures
    plt.show()
  
class BlockEstimationAnimator(animation.TimedAnimation):
    def __init__(self, plant, data, key):
        # Store the plant, data, and key
        self.plant = plant
        self.data = data
        self.key = key
        x = data[key]['state']
        # Re-create the terrain GPs
        self.height_gp, self.fric_gp = make_gp_models()
        # Create the figure
        self.fig, self.axs = plt.subplots(2,1)
        self.axs[0].set_ylabel('Terrain Height')
        self.axs[1].set_ylabel('Friction')
        self.axs[1].set_xlabel('Position')
        # Initialize the block
        self.block = Rectangle(xy=(x[0,0], x[1,0]), width=1.0, height=1.0)
        # Draw the terrain GPs
        height_prior, height_Cov = self.height_gp.prior(x[0:2,:])
        fric_prior, fric_Cov = self.fric_gp.prior(x[0:2,:])
        height_prior = np.squeeze(height_prior)
        fric_prior = np.squeeze(fric_prior)
        self.height_line = Line2D([], [], marker='x', markeredgecolor='red')
        self.fric_line = Line2D([], [], marker='x',markeredgecolor='red')
        self.height_mean = Line2D(x[0,:], height_prior,color='blue',linewidth=1.5)
        self.fric_mean = Line2D(x[0,:], fric_prior,color='blue',linewidth=1.5)
        # Draw the true terrains
        height = self.data[self.key]["height"]
        friction = self.data[self.key]["friction"]
        self.height_true = Line2D(x[0,:], height, color='green', linestyle='--',linewidth=1.5)
        self.friction_true = Line2D(x[0,:], friction, color='green', linestyle='--',linewidth=1.5)
        # Add all the lines to their axes
        self.axs[0].add_patch(self.block)
        self.axs[0].add_line(self.height_true)
        self.axs[0].add_line(self.height_line)
        self.axs[0].add_line(self.height_mean)
        self.axs[1].add_line(self.friction_true)
        self.axs[1].add_line(self.fric_line)
        self.axs[1].add_line(self.fric_mean)
        # Add in the covariances
        h_cov = np.squeeze(np.diag(height_Cov))
        f_cov = np.squeeze(np.diag(fric_Cov))
        self.axs[0].fill_between(x[0,:], height_prior - h_cov, height_prior + h_cov, alpha=0.3, color='blue')
        self.axs[1].fill_between(x[0,:], fric_prior - f_cov, fric_prior + f_cov, alpha=0.3,color='blue')
        # Set the axis limits
        self.axs[0].set_xlim(-0.5,5.5)
        self.axs[1].set_xlim(-0.5,5.5)
        self.axs[0].set_ylim(-1.0,2.0)
        self.axs[1].set_ylim(0.,1.5)
        # Setup the initial animation
        animation.TimedAnimation.__init__(self, self.fig, interval=50, repeat=False, blit=True)
    

    def _draw_frame(self, framedata):
        i = framedata
        x = self.data[self.key]['state']
        # update the block position
        xpos = x[0,i] - 0.5
        ypos = x[1,i] - 0.5
        self.block.set_xy((xpos, ypos))
        # update the height and friction data
        self.height_line.set_data(self.plant.terrain.height.data_x[0,0:i+1], self.plant.terrain.height.data_y[0,0:i+1])
        self.fric_line.set_data(self.plant.terrain.friction.data_x[0,0:i+1], self.plant.terrain.friction.data_y[0,0:i+1])
        # Update the posterior mean and covariance for height and friction
        self.height_gp.add_data(self.plant.terrain.height.data_x[:,i:i+1], self.plant.terrain.height.data_y[:,i:i+1])
        self.fric_gp.add_data(self.plant.terrain.friction.data_x[:,i:i+1], self.plant.terrain.friction.data_y[:,i:i+1])
        height_mu, height_Cov = self.height_gp.posterior(x[0:2,:])
        fric_mu, fric_Cov = self.fric_gp.posterior(x[0:2,:])
        # Update mean
        height_mu = np.squeeze(height_mu)
        fric_mu = np.squeeze(fric_mu)
        self.height_mean.set_ydata(height_mu)
        self.fric_mean.set_ydata(fric_mu)
        # Update covariance
        self.axs[0].collections.clear()
        s_height = np.squeeze(np.diag(height_Cov))
        self.axs[0].fill_between(x[0,:], height_mu - s_height, height_mu + s_height, alpha=0.3, color='blue')
        self.axs[1].collections.clear()
        s_fric = np.squeeze(np.diag(fric_Cov))
        self.axs[1].fill_between(x[0,:], fric_mu - s_fric, fric_mu + s_fric, alpha=0.3, color='blue')
        # Update the drawn artists
        self._drawn_artists = [self.block, self.height_line, self.fric_line, self.height_mean, self.fric_mean]

    def new_frame_seq(self):
        # Fix this
        return iter(range(self.data[self.key]["state"].shape[1]))

    def _init_draw(self):
        pass

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
        print(f"Estimating for {key}")
        start = timeit.default_timer()
        run_terrain_estimation(plant, t, x, u)
        stop = timeit.default_timer()
        print(f"Elapsed time {stop - start}")
        # Plot the results
        plot_terrain_results(plant, data, key)
        # Make and save an animation
        # print(f"Distilling animation for {key}")
        # ani = BlockEstimationAnimator(plant, data, key)
        # savename = key + '.mp4'
        # ani.save(savename)
        #plt.show()