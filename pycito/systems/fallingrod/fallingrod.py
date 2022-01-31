"""
FallingRod Example from Stewart and Trinkle

Luke Drnach
February 19, 2021
"""
#Library Imports
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import RigidTransform, PiecewisePolynomial
# Project Imports
from pycito.systems.timestepping import TimeSteppingMultibodyPlant
from pycito.systems.terrain import FlatTerrain
from pycito.utilities import FindResource, GetKnotsFromTrajectory
from pycito.systems.visualization import Visualizer

class FallingRod(TimeSteppingMultibodyPlant):
    def __init__(self, urdf_file="pycito/systems/fallingrod/urdf/fallingrod.urdf", terrain=FlatTerrain()):
        # Initialize the time-stepping multibody plant
        super(FallingRod, self).__init__(file=FindResource(urdf_file), terrain=terrain)
        # Weld the center body frame to the world frame
        body_inds = self.multibody.GetBodyIndices(self.model_index)
        base_frame = self.multibody.get_body(body_inds[0]).body_frame()
        self.multibody.WeldFrames(self.multibody.world_frame(), base_frame, RigidTransform())

    def plot_trajectories(self, state=None, force=None):
        if state is not None:
            self.plot_state_trajectory(state, show=False)
        if force is not None:
            self.plot_force_trajectory(force, show=False)
        if state is not None or force is not None:
            plt.show()

    def plot_force_trajectory(self, force, show=True):
        t, f = GetKnotsFromTrajectory(force)
        _, axs = plt.subplots(3,1)
        f = self.resolve_forces(f)
        labels = ['Normal', 'Friction-X', 'Friction-Y']
        for k in range(0,2):
            for n in range(0, 3):
                axs[n].plot(t, f[2*n+k, :], linewidth=1.5)
                axs[n].set_ylabel(labels[n])
        axs[0].set_title("Reaction Forces")
        axs[-1].set_xlabel('Time (s)')
        if show:
            plt.show()

    @staticmethod
    def plot_state_trajectory(state, show=True):
        t, x = GetKnotsFromTrajectory(state)
        # Position plot
        _, axs = plt.subplots(3,1)
        labels = ['Horizontal', 'Vertical', 'Rotation']
        for n in range(0, 3):
            axs[n].plot(t,x[n,:], linewidth=1.5)
            axs[n].set_ylabel(labels[n])
        axs[-1].set_xlabel('Time (s)')
        axs[0].set_title('Position')
        # Velocity plot
        _, axs2 = plt.subplots(3,1)
        for n in range(0, 3):
            axs2[n].plot(t,x[3+n,:], linewidth=1.5)
            axs2[n].set_ylabel(labels[n])
        axs2[-1].set_xlabel('Time (s)')
        axs2[0].set_title('Velocity')
        if show:
            plt.show()

    @staticmethod
    def visualize(trajectory=None):
        vis = Visualizer("pycito/systems/fallingrod/urdf/fallingrod.urdf")
        #Weld the center body frame to the world frame
        body_inds = vis.plant.GetBodyIndices(vis.model_index)
        base_frame = vis.plant.get_body(body_inds[0]).body_frame()
        vis.plant.WeldFrames(vis.plant.world_frame(), base_frame, RigidTransform())
        # Make the visualization
        vis.visualize_trajectory(trajectory)

def falling_rod_main():
    # Create a falling rod example
    terrain = FlatTerrain(height=0., friction=0.6)
    rod = FallingRod(terrain=terrain)
    rod.Finalize()
    # Initial conditions:
    x0 = np.array([0., 1., np.pi/3, 0., 0., 4.])
    dt = 0.01
    # Simulate
    t, x, f = rod.simulate(dt, x0, N=100)
    state = PiecewisePolynomial.FirstOrderHold(t, x)
    force = PiecewisePolynomial.FirstOrderHold(t, f)
    # Plot the results
    rod.plot_trajectories(state, force)
    # Visualize
    rod.visualize(state)

if __name__ == '__main__':
    print("hello from fallingrod.py!")
