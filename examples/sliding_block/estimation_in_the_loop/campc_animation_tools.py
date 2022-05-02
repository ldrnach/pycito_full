import os, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from pycito.controller.optimization import OptimizationLogger
from pycito.controller.mpc import ReferenceTrajectory

# TODO: Debugging

# Globals
LOADDIR = os.path.join('examples','sliding_block','estimation_in_the_loop')
MPCFILE = os.path.join('mpc_logs','mpclogs.pkl')
CAMPCFILE = os.path.join('campc_logs','mpclogs.pkl')
REFTRAJ = os.path.join('data','slidingblock','block_reference.pkl')

class BlockMPCAnimator():
    def __init__(self, truemodel, reftraj):
        self.truemodel = truemodel
        self.reftraj = reftraj
        self._contact_samples = None
        # Video assets
        self.block = None
        self.plan = None
        self.normals = None
        self.friction = None
   
    def _make_figure(self):
        """Initialize the figure for the animations"""
        self.fig = plt.figure()
        self.axs = self.fig.subplot_mosaic(
            [
                ['main'],
                ['main'],
                ['main'],
                ['friccoeff'],
                ['normal_force'],
                ['tangent_force']
            ]
        )
        self.fig.canvas.manager.full_screen_toggle()
        self.axs['main'].set_ylabel('Motion')
        self.axs['friccoeff'].set_ylabel('Fric\nCoeff')
        self.axs['normal_force'].set_ylabel('Normal')
        self.axs['tangent_force'].set_ylabel('Friction')

    def set_contact_samples(self, logs, sampling=200):
        """Create a set of dummy points for evaluating and drawing the contact model"""
        initial_pos = np.array([log['initial_state'][0] for log in logs])
        x0, xf = np.zeros((3,)), np.zeros((3,))
        x0[0], xf[0] = np.amin(initial_pos), np.amax(initial_pos) + 0.5
        self._contact_samples = np.linspace(x0, xf, sampling).transpose()

    def get_contact_model_points(self, model):
        """Return the drawing points for the contact model"""
        surf = model.find_surface_zaxis_zeros(self._contact_samples)
        fric = np.concatenate([model.eval_friction(pt) for pt in self._contact_samples.T], axis=0)
        return surf, fric

    def draw_contact_model(self, model, label, linecolor):
        """Draw the surface and friction coefficients of the contact model"""
        surf, fric = self.get_contact_model_points(model)
        self.axs['main'].plot(surf[0,:], surf[2,:], linewidth=1.5, color=linecolor, label=label)
        self.axs['friccoeff'].plot(surf[0,:], fric, linewidth=1.5, color=linecolor, label=label)

    def setup(self, logs):
        """Setup the animation"""
        # Draw the contact models
        self.set_contact_samples(logs)
        self.draw_contact_model(self.truemodel.terrain, label='True Terrain', linecolor='k')
        self.draw_contact_model(self.reftraj.plant.terrain, label='Expected Terrain', linecolor='b')
        # Make the block
        self.setup_block(logs[0])
        # Setup the motion plan and forces
        self.setup_plan(logs[0], color='g', label='MPC Plan')
        self.setup_forces(logs[0], color='g', label='MPC Forces')      
        # # Add the legends
        # self.axs['main'].legend(handles = [self.plan], loc='upper left', ncol=1, frameon=False)
        # self.axs['friccoeff'].legend(loc='upper left', ncol=2, frameon=False)
        # self.axs['normal_force'].legend(loc='lower right', ncol=2, frameon=False)

    def setup_block(self, log):
        """Create a Rectangle to represent the block"""
        x0 = log['initial_state']
        self.block = self.axs['main'].add_patch(Rectangle((x0[0] - 0.5, x0[1] - 0.5), 1, 1))

    def _get_plan(self, log):
        """Re-create the motion plan in the original, state-vector format"""
        t = log['time']
        dx = log['state']
        horizon = dx.shape[1]
        start = self.reftraj.getTimeIndex(t)
        refstates = np.column_stack([self.reftraj.getState(idx) for idx in range(start,start + horizon)])
        return refstates + dx

    def _get_forces(self, log):
        """Recreate the force component of the motion plan"""
        # Position information
        state = self._get_plan(log)
        # Reaction forces
        force = log['force']
        fN = force[0,:]
        fT = force[1,:] - force[3,:]    # X-axis forces only
        return state[0,1:], fN, fT

    def setup_plan(self, log, color, label):
        """Create the line representing the motion plan"""
        plan = self._get_plan(log)
        self.plan,  = self.axs['main'].plot(plan[0,:], plan[1,:], linewidth=1.5, color=color, label=label)

    def setup_forces(self, log, color, label):
        """Setup the force lines for the motion plan"""
        # Setup the reference forces
        x0 = self.reftraj._state[0,:]
        fN = self.reftraj._force[0,:]
        fT = self.reftraj._force[1,:] - self.reftraj._force[3,:]
        self.axs['normal_force'].plot(x0, fN, linewidth=1.5, label='Expected', color='k')
        self.axs['tangent_force'].plot(x0, fT, linewidth=1.5, label='Expected', color='k')
        # Setup the initial motion plan forces
        x, fN, fT = self._get_forces(log)
        self.normals, = self.axs['normal_force'].plot(x, fN, linewidth=1.5, color= color, label=label)
        self.friction, = self.axs['tangent_force'].plot(x, fT, linewidth=1.5, color= color, label=label)
    
    def get_artists(self):
        """Return the updatable artists"""
        return self.block, self.plan, self.normals, self.friction

    def update(self, log):
        """Update the animation frame"""
        self.update_block_position(log)
        self.update_motion_plan(log)
        self.update_forces(log)
        return self.get_artists()

    def update_block_position(self, log):
        """Update the block position"""
        x0 = log['initial_state']
        self.block.set(x = x0[0] - 0.5, y = x0[1] - 0.5)

    def update_motion_plan(self, log):
        """Update the motion plan"""
        plan = self._get_plan(log)
        self.plan.set(xdata = plan[0,:], ydata=plan[1,:])

    def update_forces(self, log):
        """Update the MPC force trace"""
        x, fN, fT = self._get_forces(log)
        self.normals.set(xdata = x, ydata = fN)
        self.friction.set(xdata = x, ydata = fT)        

    def animate(self, mpclogs, savename):
        """Create an animation using the MPC log files"""
        self._make_figure()
        self.setup(mpclogs)
        print(f"Distilling MPC Animation")
        anim = FuncAnimation(self.fig, 
                            self.update, 
                            init_func = self.get_artists,
                            frames = mpclogs,
                            interval = 50,
                            blit = True)
        anim.save(savename, writer='ffmpeg')
        print(f"Finished! MPC Animation saved to {savename}")

class BlockCAMPCComparisonAnimator(BlockMPCAnimator):
    def __init__(self, truemodel, reftraj, esttraj):
        super().__init__(truemodel, reftraj)
        self.esttraj = esttraj
        # Additional video assets
        self.surfline = None
        self.fricline = None
        self.campc_block = None
        self.campc_plan = None
        self.campc_normals = None
        self.campc_friction = None
        self.contact_pt = None

    def get_updated_contact_model(self, log):
        """Return a updated copy of the contact model, using the values in the log"""
        start = self.esttraj.getTimeIndex(log['time'])
        stop = start + log['normal_forces'].shape[1]
        cpts = np.column_stack(self.esttraj.get_contacts(start, stop))
        model = copy.deepcopy(self.esttraj.contact_model)
        fc_weights = self._variables_to_friction_weights(log['friction_weights'], log['normal_forces'])
        model.add_samples(cpts, log['distance_weights'], fc_weights)
        return model

    def _variables_to_friction_weights(self, fweights, forces):
        """
            Calculate the friction coefficient weights from solution variables
        """
        fN = forces.reshape(-1)
        fc_weights = np.zeros_like(fweights)
        err_index = fN > self.esttraj._FTOL 
        if np.any(err_index):
            fc_weights[err_index] = fweights[err_index]/fN[err_index]
        return fc_weights

    def update_contact_model_lines(self, log):
        """Update the estimated contact model"""
        # Copy and Update the contact model first
        est_model = self.get_updated_contact_model(log)
        # Update the lines
        surf, fric = self.get_contact_model_points(est_model)
        self.surfline.set(xdata=surf[0,:], ydata=surf[2,:])
        self.fricline.set(xdata=surf[0,:], ydata=fric)
        # Update the contact model marker
        x0 = np.zeros((3,1))
        x0[0,0], x0[2,0] = log['initial_state'][0], log['initial_state'][1]
        self.contact_pt.set(xdata = x0[0], ydata = est_model.eval_friction(x0))
    
    def update_motion_plan(self, mpclog, campclog):
        """Update the CAMPC and MPC motion plans"""
        super().update_motion_plan(mpclog)
        plan = self._get_plan(campclog)
        self.campc_plan.set(xdata = plan[0,:], ydata = plan[1,:])

    def update_forces(self, mpclog, campclog):
        """Update the MPC and CAMPC force traces"""
        super().update_forces(mpclog)
        x, fN, fT = self._get_forces(campclog)
        self.campc_normals.set(xdata = x, ydata = fN)
        self.campc_friction.set(xdata = x, ydata = fT)

    def update_block_position(self, mpclog, campclog):
        """Update the block position for MPC and CAMPC"""
        super().update_block_position(mpclog)
        x0 = campclog['initial_state']
        self.campc_block.set(x = x0[0] - 0.5, y = x0[1] - 0.5)
        self.contact_pt.set(xdata = x0[0], ydata = x0[1] - 0.5)

    def setup(self, mpclogs, campclog):
        """Setup the Animation Window"""
        super().setup(mpclogs)
        # Draw the estimated terrain
        self.setup_estimated_contact(self.esttraj.contact_model, campclog[0], label='Estimated Terrain', color='y')
        # Make the block
        self.setup_estimated_block(campclog[0])
        # Setup the motion plan and forces
        self.setup_estimated_plan(campclog[0], color='orange', label='CAMPC Plan')
        self.setup_estimated_forces(campclog[0], color='orange', label='CAMPC Forces')
        # Add the legends
        # self.axs['main'].legend(handles = [self.plan, self.campc_plan], loc='upper left', ncol=2, frameon=False)
        # self.axs['friccoeff'].legend(loc='upper left', ncol=3, frameon=False)
        # self.axs['normal_force'].legend(loc='lower right', ncol=2, frameon=False)

    def setup_estimated_contact(self, model, log, label, color):
        """Draw the surface and friction coefficients of the contact model"""
        surf, fric = self.get_contact_model_points(model)
        self.surfline, = self.axs['main'].plot(surf[0,:], surf[2,:], linewidth=1.5, color=color, label=label)
        self.fricline, = self.axs['friccoeff'].plot(surf[0,:], fric, linewidth=1.5, color=color, label=label)
        x0 = np.zeros((3,1))
        x0[0,0], x0[2,0] = log['initial_state'][0], log['initial_state'][1]
        self.contact_pt,  = self.axs['friccoeff'].plot(x0[0], model.eval_friction(x0), marker='o', markeredgecolor='orange', markerfacecolor='orange')

    def setup_estimated_block(self, log):
        """Create a Rectangle to represent the block"""
        x0 = log['initial_state']
        self.campc_block = self.axs['main'].add_patch(Rectangle((x0[0] - 0.5, x0[1] - 0.5), 1, 1))
        

    def setup_estimated_plan(self, log, color, label):
        """Create the line representing the motion plan"""
        plan = self._get_plan(log)
        self.campc_plan, = self.axs['main'].plot(plan[0,:], plan[1,:], linewidth=1.5, color=color, label=label)

    def setup_estimated_forces(self, log, color, label):
        """Create the line representing the replanned forces"""
        # Setup the initial motion plan forces
        x, fN, fT = self._get_forces(log)
        self.campc_normals, = self.axs['normal_force'].plot(x, fN, linewidth=1.5, color= color, label=label)
        self.campc_friction, = self.axs['tangent_force'].plot(x, fT, linewidth=1.5, color= color, label=label)

    def get_artists(self):
        """Return the updatable artists for animation"""
        mpc_artists = super().get_artists()
        return (*mpc_artists, self.surfline, self.fricline, self.campc_block, self.campc_plan, self.campc_normals, self.campc_friction, self.contact_pt)

    def update(self, logs):
        """Update the animation"""
        mpc, campc = logs
        self.update_block_position(mpc, campc)
        self.update_forces(mpc, campc)
        self.update_motion_plan(mpc, campc)
        self.update_contact_model_lines(campc)
        return self.get_artists()

    def animate(self, mpclogs, campclogs, savename):
        """Create an animation using the MPC log files"""
        logs = zip(mpclogs, campclogs)
        self._make_figure()
        self.setup(mpclogs, campclogs)
        print(f"Distilling CAMPC Animation")
        anim = FuncAnimation(self.fig, 
                            self.update, 
                            init_func = self.get_artists,
                            frames = logs,
                            interval = 50,
                            blit = True)
        anim.save(savename, writer='ffmpeg')
        print(f'Finished! CAMPC Animation saved at {savename}')

if __name__ == '__main__':
    print("Hello from campc_animation_tools!")