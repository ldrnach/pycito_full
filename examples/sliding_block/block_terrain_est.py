"""
Terrain estimation with the sliding block data

Luke Drnach
November 17, 2020
"""
import timeit
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import utilities as utils
import systems.gaussianprocess as gp
from systems.terrainestimator import ResidualTerrainEstimator, ResidualTerrainEstimation_Debug
from systems.terrain import GaussianProcessTerrain
from systems.block.block import Block

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

def run_terrain_estimation_debug(plant, t, x, u):
    """run residual terrain estimation in debug mode """
    estimator = ResidualTerrainEstimation_Debug(plant)
    for n in range(0, x.shape[1]-1):
        h = t[n+1]-t[n]
        estimator.estimate_terrain(h, x[:,n], x[:,n+1], u[:,n])
    # Print out the debug report
    estimator.print_report()
    # Plot the constraint violations
    estimator.plot_constraints()

# def run_terrain_estimation_debug(plant, t, x, u):
#     """ runs residual terrain estimation and updates the plant"""
#     estimator = ResidualTerrainEstimator(plant)
#     soln1 = {"dist_err": [],
#                 "fric_err": [],
#                 "fN": [],
#                 "fT": [],
#                 "gam": [],
#                 "success": [],
#                 "solver": [],
#                 "status": [],
#                 "infeasible":[]
#         }
#     soln2 = {key: value[:] for key, value in soln1.items()}
#     for n in range(0, x.shape[1]-1):
#         h = t[n+1]-t[n]
#         soln1_t, soln2_t = estimator.estimate_terrain(h, x[:,n], x[:,n+1], u[:,n])
#         soln1 = append_entries(soln1, soln1_t)
#         soln2 = append_entries(soln2, soln2_t)
#     for key in soln1.keys():
#         if key == "infeasible":
#             soln1[key] = [name for name in soln1[key] if name]
#             soln2[key] = [name for name in soln1[key] if name]
#         elif key != "success" and key != 'solver' and key != "status":
#             soln1[key] = np.concatenate(soln1[key], axis=1)
#             soln2[key] = np.concatenate(soln2[key], axis=1)

#     return soln1, soln2

def append_entries(target_dict, source_dict):
    for key in target_dict.keys():
        if key == "success" or key == "solver" or key == "status" or key == "infeasible":
            target_dict[key].append(source_dict[key])
        else:
            target_dict[key].append(np.expand_dims(source_dict[key], axis=1))
    return target_dict

def plot_debug_results(soln1, soln2, data):
    t = data["time"]
    ts = t[0:-1]
    f = data["force"]
    x = data["state"]
    # Check for samples that did not solve appropriately
    mask1 = np.invert(np.array(soln1["success"]))
    mask2 = np.invert(np.array(soln2["success"]))
    t_false1 = ts[mask1]
    t_false2 = ts[mask2]
    dt = t[1]
    _, axs = plt.subplots(3,1)
    axs[0].plot(ts, soln1["dist_err"][0,:], linewidth=1.5, label="Pass 1")
    axs[0].plot(ts, soln2["dist_err"][0,:], linewidth=1.5, label="Pass 2")
    axs[0].set_ylabel("Height residuals")
    axs[1].plot(ts, soln1["fric_err"][0,:], linewidth=1.5, label="Pass 1")
    axs[1].plot(ts, soln2["fric_err"][0,:], linewidth=1.5, label="Pass 2")
    axs[1].set_ylabel("Friction residuals")
    axs[2].plot(ts, soln1["gam"][0,:], linewidth=1.5, label="Pass 1")
    axs[2].plot(ts, soln2["gam"][0,:], linewidth=1.5, label="Pass 2")
    axs[2].plot(t, x[2,:], linewidth=1.5, label="Sim")
    axs[2].set_ylabel("Velocity")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()
    add_solve_highlights(axs, t_false1, t_false2, dt)
    _, axs2 = plt.subplots(3,1)
    # Normal force subplot
    axs2[0].plot(ts, soln1["fN"][0,:], linewidth=1.5, label="Pass 1")
    axs2[0].plot(ts, soln2["fN"][0,:], linewidth=1.5, label="Pass 2")
    axs2[0].plot(t, f[0,:], linewidth=1.5, label="Sim")
    axs2[0].set_ylabel('Normal force')
    # Friction force subplot
    axs2[1].plot(ts, soln1["fT"][0,:] - soln1["fT"][2,:], linewidth=1.5, label="Pass 1")
    axs2[1].plot(ts, soln2["fT"][0,:] - soln2["fT"][2,:], linewidth=1.5, label="Pass 2")
    axs2[1].plot(t, f[1,:] - f[3,:], linewidth=1.5, label="Sim")
    axs2[1].set_ylabel('X Friction Force')
    axs2[2].plot(ts, soln1["fT"][1,:] - soln1["fT"][3,:], linewidth=1.5, label="Pass 1")
    axs2[2].plot(ts, soln2["fT"][1,:] - soln1["fT"][3,:], linewidth=1.5, label="Pass 2")
    axs2[2].plot(t, f[2,:] - f[4,:], linewidth = 1.5, label="Sim")
    axs2[2].set_ylabel("Y Friction Force")
    axs2[2].legend()
    add_solve_highlights(axs2, t_false1, t_false2, dt)
    plt.show()

def add_solve_highlights(axs, t_false1, t_false2, dt):
    for k in range(0, len(axs)):
        for n in range(0, len(t_false1)):
            axs[k].axvspan(t_false1[n], t_false1[n]+dt, color="tab:blue", alpha=0.5, zorder=1)
        for n in range(0, len(t_false2)):
            axs[k].axvspan(t_false2[n], t_false2[n] + dt, color="tab:orange", alpha=0.5, zorder=1)



def plot_terrain_results(plant, data):
    """plot the results from terrain estimation"""
    x = data["state"]
    h = data["height"]
    m = data["friction"]
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
        run_terrain_estimation_debug(plant, t, x, u)
        stop = timeit.default_timer()
        # print(f"Elapsed time {stop - start}")
        # print(f"Pass 1 solvers used: {set(soln1['solver'])}")
        # print(f"Pass 1 terminated with exit codes: {set(soln1['status'])}")
        # print(f"Pass 1 infeasible constraints: {soln1['infeasible']}")
        # print(f"Pass 2 solvers used: {set(soln2['solver'])}")
        # print(f"Pass 2 terminated with exit codes: {set(soln2['status'])}")
        # print(f"Pass 2 infeasible constraints: {soln2['infeasible']}")
        #plot_debug_results(soln1, soln2, data[key])
        # Plot the results
        #plot_terrain_results(plant, data[key])
        # Make and save an animation
        # print(f"Distilling animation for {key}")
        # ani = BlockEstimationAnimator(plant, data, key)
        # savename = key + '.mp4'
        # ani.save(savename)
        # plt.show()