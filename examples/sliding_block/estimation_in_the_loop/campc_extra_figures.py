import os
import numpy as np
import matplotlib.pyplot as plt
import pycito.utilities as utils
from pycito.controller.optimization import OptimizationLogger


REFSOURCE = os.path.join('data','slidingblock','block_reference.pkl')
SOURCE = os.path.join('data','estimationcontrol','final_for_paper','final_for_paper')
PARTS = ['flatterrain','high_friction','low_friction','stepterrain']
MPC_SIM = 'mpcsim.pkl'
CPC_SIM = 'campcsim.pkl'
MPC_LOGS = os.path.join('mpc_logs','mpclogs.pkl')
CAMPC_LOGS = os.path.join('campc_logs','mpclogs.pkl')
EXT = '.pdf'

FORCE_MATRIX = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0]])

def get_forces_from_logs(logs):
    force_vars = np.column_stack([log['force'][:, 0] for log in logs])
    return FORCE_MATRIX.dot(force_vars)   

def do_trajectory_plotting():
    ref = utils.load(REFSOURCE)
    for part in PARTS:
        mpc_data = utils.load(os.path.join(SOURCE, part, MPC_SIM))
        campc_data = utils.load(os.path.join(SOURCE, part, CPC_SIM))
        fig, axs = plt.subplots(3,1)
        dataset = [ref, mpc_data, campc_data]
        labels = ['Reference','MPC','CAMPC']
        for data, label in zip(dataset, labels):
            axs[0].plot(data['time'], data['state'][0,:], linewidth=1.5, label=label)   #Position
            axs[1].plot(data['time'], data['state'][2,:], linewidth=1.5, label=label)   #Velocity
            axs[2].plot(data['time'], data['control'][0,:], linewidth=1.5, label=label) #Control
        axs[0].set_ylabel('Position (m)')
        axs[1].set_ylabel('Velocity (m/s)')
        axs[2].set_ylabel('Control (N)')
        axs[2].set_xlabel('Time (s)')
        axs[0].legend(frameon=False)
        axs[0].set_title(part)
        for k in range(3):
            axs[k].set_xlim([0,1.5])
            fig.tight_layout()
        fig.savefig(os.path.join(SOURCE,part, 'horizontal_traj_with_reference' + EXT), dpi=fig.dpi, bbox_inches='tight')
        print(f"{part} figure saved!")    
    
def do_logscale_force_plots():
    part = 'stepterrain'
    mpc_logs = OptimizationLogger.load(os.path.join(SOURCE, part, MPC_LOGS)).logs
    cpc_logs = OptimizationLogger.load(os.path.join(SOURCE, part, CAMPC_LOGS)).logs
    mpc_pred_force = get_forces_from_logs(mpc_logs)
    cpc_pred_force = get_forces_from_logs(cpc_logs)
    mpcdata = utils.load(os.path.join(SOURCE, part, MPC_SIM))
    cpcdata = utils.load(os.path.join(SOURCE, part, CPC_SIM))
    mpc_force = FORCE_MATRIX.dot(mpcdata['force'][:,1:])
    cpc_force = FORCE_MATRIX.dot(cpcdata['force'][:,1:])
    # Plot the individual force traces
    fig2, axs2 = plt.subplots(3,1)
    fig3, axs3 = plt.subplots(3,1)
    t = mpcdata['time'][1:]
    for n, name in enumerate(['Normal','Friction-X','Friction-Y']):
        axs2[n].plot(t, mpc_force[n,:], linewidth=1.5, label='Simulation')
        axs2[n].plot(t, mpc_pred_force[n,:], linewidth=1.5, label='Predicted')
        axs3[n].plot(t, cpc_force[n,:], linewidth=1.5, label='Simulation')
        axs3[n].plot(t, cpc_pred_force[n,:], linewidth=1.5, label='Predicted')
        axs2[n].set_ylabel(name)
        axs3[n].set_ylabel(name)
        axs2[n].set_yscale('symlog', linthresh=1)
        axs3[n].set_yscale('symlog', linthresh=1)
        axs2[n].grid()
        axs3[n].grid()
    axs2[-1].set_xlabel('Time (s)')
    axs2[0].set_title(f'{part} MPC Force Mismatch')
    axs3[-1].set_xlabel('Time (s)')
    axs3[0].set_title(f'{part} CAMPC Force Mismatch')
    axs2[0].legend(frameon=False)
    axs3[0].legend(frameon=False)
    plt.show()
    fig2.savefig(os.path.join(SOURCE, part + '_MPC_Forces_logscale' + EXT), dpi=fig2.dpi, bbox_inches='tight')
    fig3.savefig(os.path.join(SOURCE, part + '_CAMCP_Forces_logscale' + EXT), dpi=fig3.dpi, bbox_inches='tight')
    print(f"\tSaved figures for {part}")

if __name__ == '__main__':
    do_logscale_force_plots()