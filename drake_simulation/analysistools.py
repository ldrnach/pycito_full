import os
import numpy as np
import matplotlib.pyplot as plt
from pycito.systems.A1.a1 import A1VirtualBase
import pycito.decorators as deco

from pydrake.all import PiecewisePolynomial as pp
LOGNAME = 'mpclogs.pkl'
LOGFIGURENAME = 'mpclogs'
EXT = '.png'
class _A1ErrorPlotter(A1VirtualBase):
    def __init__(self):
        super().__init__()
        self.Finalize()

    @deco.showable_fig
    @deco.saveable_fig
    def plot_individual_tracking_errors(self, simdata, reftraj):
        """Plot the tracking errors"""
        # calculate the state error
        state_error = calculate_tracking_error(simdata, reftraj)
        time = simdata['time'][1:]
        state_error = state_error[:, 1:]
        all_fig = []
        all_axs = []
        q, v = np.split(state_error, [self.multibody.num_positions()])
        # Base position
        fig, axs = self._plot_base_position(time, q[[3, 4, 5, 0, 1, 2], :], show=False, savename=None)
        axs[0].set_title('Base Position Tracking Error')
        all_fig.append(fig)
        all_axs.append(axs)
        # Joint position
        fig, axs = self._plot_joint_position(time, q[6:], show=False, savename=None)
        axs[0].set_title('Joint Position Tracking Error')
        all_fig.append(fig)
        all_axs.append(axs)
        # Base Velocity
        fig, axs = self._plot_base_velocity(time, v[[3, 4, 5 ,0 , 1, 2], :], show=False, savename=None)
        axs[0].set_title('Base Velocity Tracking Error')
        all_fig.append(fig)
        all_axs.append(axs)
        # Joint Velocity
        fig, axs = self._plot_joint_velocity(time, v[6:,:], show=False, savename=None)
        axs[0].set_title('Joint Velocity Tracking Error')
        all_fig.append(fig)
        all_axs.append(axs)
        return all_fig, all_axs
    
    @deco.showable_fig
    @deco.saveable_fig
    def plot_tracking_error_summary(self, simdata, reftraj):
        """
        Plot the error accumulated over all coordinates for the base and joints separately
        """
        # First, calculate the state erorr
        state_error = calculate_tracking_error(simdata, reftraj)
        time = simdata['time']
        time = time[1:]
        state_error = state_error[:, 1:]

        q, v = np.split(state_error, [self.multibody.num_positions()])
        base_pos_mse = np.sqrt(np.mean(q[:6,:]**2, axis=0))
        joint_pos_mse = np.sqrt(np.mean(q[6:, :]**2, axis=0))
        base_vel_mse = np.sqrt(np.mean(v[:6,:]**2, axis=0))
        joint_vel_mse = np.sqrt(np.mean(v[6:,:]**2, axis=0))
        # Make the plot
        fig, axs = plt.subplots(3,1)
        axs[0].plot(time, base_pos_mse, linewidth=1.5, label='Base')
        axs[0].plot(time, joint_pos_mse, linewidth=1.5, label='Joint')
        axs[0].set_ylabel('Position Error')
        axs[0].legend(frameon=False, ncol=2)
        axs[0].set_title('Root-Mean-Squared Tracking Error')
        axs[1].plot(time, base_vel_mse, linewidth=1.5, label='Base')
        axs[1].plot(time, joint_vel_mse, linewidth=1.5, label='Joint')
        axs[1].set_ylabel('Velocity Error')
        # Calculate the foot error
        refstate_traj = pp.ZeroOrderHold(reftraj._time, reftraj._state)
        refstate = refstate_traj.vector_values(simdata['time']) 
        ref_feet = self.state_to_foot_trajectory(refstate)
        sim_feet = self.state_to_foot_trajectory(simdata['state'])
        labels = ['FR','FL','BR','BL']
        for ref_foot, sim_foot, label in zip(ref_feet, sim_feet, labels):
            foot_err = np.sqrt(np.mean((ref_foot - sim_foot)**2, axis=0))
            axs[2].plot(time, foot_err[1:], linewidth=1.5, label=label)
        axs[2].set_ylabel('Foot Position\nError')
        axs[2].set_xlabel('Time (s)')
        axs[2].legend(frameon=False, ncol=4)
        return fig, axs

def calculate_tracking_error(simdata, reftraj):
    """Calculate the tracking error between simulation and reference trajectory"""
    # Resample the reference trajectory at the simulation sample times
    ref_state_traj = pp.ZeroOrderHold(reftraj._time, reftraj._state)
    ref_state = ref_state_traj.vector_values(simdata['time'])
    sim_state = simdata['state']
    # The error
    return sim_state - ref_state


def plot_tracking_error(simdata, reftraj, savedir):
    plotter = _A1ErrorPlotter()
    plotter.plot_individual_tracking_errors(simdata, reftraj, show=False, savename=os.path.join(savedir, 'individual_tracking_errors'+EXT))
    plotter.plot_tracking_error_summary(simdata, reftraj, show=False, savename=os.path.join(savedir, 'tracking_error_summary'+EXT))

def save_mpc_logs(controller, savedir):
    print("Saving MPC logs...", end="", flush=True)
    controller.logger.save(os.path.join(savedir, LOGNAME))
    print("\tDone!")

def plot_mpc_logs(controller, savedir=None):
    print("Plotting MPC logs...", end="", flush=True)
    controller.logger.plot(show=False, savename=os.path.join(savedir, LOGFIGURENAME + EXT))
    print('\tDone!')