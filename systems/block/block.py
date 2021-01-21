"""
Classes and methods for creating and visualizing the sliding block

Luke Drnach
January 14, 2021
"""
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import RigidTransform, PiecewisePolynomial
# Project-specific imports
from utilities import FindResource, GetKnotsFromTrajectory
from systems.timestepping import TimeSteppingMultibodyPlant
from systems.visualization import Visualizer
from systems.terrain import FlatTerrain
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

class Block(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="systems/block/urdf/sliding_block.urdf", terrain=FlatTerrain()):

        # Initialize the time-stepping multibody plant
        super(Block, self).__init__(file=FindResource(urdf_file), terrain=terrain)
        # Weld the center body frame to the world frame
        body_inds = self.multibody.GetBodyIndices(self.model_index)
        base_frame = self.multibody.get_body(body_inds[0]).body_frame()
        self.multibody.WeldFrames(self.multibody.world_frame(), base_frame, RigidTransform())

    def visualize_pyplot(self, trajectory=None):
        if type(trajectory) is np.ndarray:
            t = np.linspace(start=0, stop=1, num=trajectory.shape[1])
            trajectory = PiecewisePolynomial.FirstOrderHold(t, trajectory)
        if type(trajectory) is not PiecewisePolynomial:
            raise ValueError("trajectory must be a PiecewisePolynomial or 2D numpy array")
        return BlockPyPlotAnimator(self, trajectory)
        
    def plot_trajectory(self, xtraj=None, utraj=None, ftraj=None):
        """
        plot the state, control, and force trajectories for the Block
        
        Arguments:
            xtraj, utraj, and ftraj should be pyDrake PiecewisePolynomials
        """
        #TODO: generalize for multiple contact points, generalize for free rotation
        # Plot the State trajectories
        if xtraj is not None:
            t, x = GetKnotsFromTrajectory(xtraj)
            fig1, axs1 = plt.subplots(2,1)
            axs1[0].plot(t, x[0,:], linewidth=1.5, label='horizontal')
            axs1[0].plot(t, x[1,:], linewidth=1.5, label='vertical')
            axs1[0].set_ylabel('Position (m)')
            axs1[1].plot(t, x[2,:], linewidth=1.5, label='horizontal')
            axs1[1].plot(t, x[3,:], linewidth=1.5, label='vertical')
            axs1[1].set_ylabel('Velocity (m/s)')
            axs1[1].set_xlabel('Time (s)')
            axs1[0].legend()
        # Plot the controls
        if utraj is not None:
            t, u = GetKnotsFromTrajectory(utraj)
            fig2, axs2 = plt.subplots(2,1)
            axs2[0].plot(t, u, linewidth=1.5)
            axs2[0].set_ylabel('Control (N)')
            axs2[0].sey_xlabel('Time (s)')
        # Plot the reaction forces
        if ftraj is not None:
            t, f = GetKnotsFromTrajectory(ftraj)
            fig3, axs3 = plt.subplots(3,1)
            axs3[0].plot(t, f[0,:], linewidth=1.5)
            axs3[0].set_ylabel('Normal')
            axs3[0].set_title('Ground reaction forces')
            axs3[1].plot(t, f[1,:] - f[3,:], linewidth=1.5)
            axs3[1].set_ylabel('Friction-x')
            axs3[2].plot(t, f[2, :] - f[4,:], linewidth=1.5)
            axs3[2].set_ylabel('Friction-y')
            axs3[2].set_xlabel('Time (s)')
        # Show the plots only when one of the inputs is not None
        if xtraj is not None or utraj is not None or ftraj is not None:
            plt.show()

    @staticmethod
    def visualize(trajectory=None):
        vis = Visualizer("systems/block/urdf/sliding_block.urdf")
        #Weld the center body frame to the world frame
        body_inds = vis.plant.GetBodyIndices(vis.model_index)
        base_frame = vis.plant.get_body(body_inds[0]).body_frame()
        vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())
        # Make the visualization
        vis.visualize_trajectory(trajectory)

class BlockPyPlotAnimator(animation.TimedAnimation):
    #TODO: Calculate viewing limits from trajectory
    #TODO: Get height and width of the block from the plant
    def __init__(self, plant, xtraj):
        # Store the plant, data, and key
        self.plant = plant
        _, x = GetKnotsFromTrajectory(xtraj)
        self.xtraj = x
        # Calculate the terrain height along the trajectory
        height = np.zeros((x.shape[1],))
        for n in range(0,x.shape[1]):
            pt = plant.terrain.nearest_point(x[0:3,n])
            height[n] = pt[2]
        # Create the figure
        self.fig, self.axs = plt.subplots(2,1)
        self.axs[0].set_ylabel('Terrain Height')
        self.axs[1].set_ylabel('Friction')
        self.axs[1].set_xlabel('Position')
        # Initialize the block
        self.block = Rectangle(xy=(x[0,0], x[1,0]), width=1.0, height=1.0)
        # Draw the true terrains
        self.height_true = Line2D(x[0,:], height, color='black', linestyle='-',linewidth=2.0)
        # Add all the lines to their axes
        self.axs[0].add_patch(self.block)
        self.axs[0].add_line(self.height_true)
        # Set the axis limits
        self.axs[0].set_xlim(-0.5,5.5)
        self.axs[0].set_ylim(-1.0,2.0)
        # Setup the initial animation
        animation.TimedAnimation.__init__(self, self.fig, interval=50, repeat=False, blit=True)
    
    def _draw_frame(self, framedata):
        i = framedata
        # update the block position
        xpos = self.xtraj[0,i] - 0.5
        ypos = self.xtraj[1,i] - 0.5
        self.block.set_xy((xpos, ypos))
        # Update the drawn artists
        self._drawn_artists = [self.block]

    def new_frame_seq(self):
        # Fix this
        return iter(range(self.xtraj.shape[1]))

    def _init_draw(self):
        pass

if __name__ == "__main__":
    block = Block()
    block.visualize()        