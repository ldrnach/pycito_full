"""
Runs analysis for the campc simulations

"""
import os, csv
import numpy as np
import matplotlib.pyplot as plt

from pycito.controller.optimization import OptimizationLogger
import pycito.utilities as utils

SOURCE = os.path.join('examples','sliding_block','estimation_in_the_loop','final')
PARTS = ['flatterrain','stepterrain','high_friction','low_friction']
CPC_SIM = 'campcsim.pkl'
MPC_SIM = 'mpcsim.pkl'

CPC_LOG = os.path.join('campc_logs','mpclogs.pkl')
MPC_LOG = os.path.join('mpc_logs','mpclogs.pkl')

REFDATA = os.path.join('data','slidingblock','block_reference.pkl')

FORCE_MATRIX = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0]])

def write_to_file(filename, rowdata):
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        for row in rowdata:
            writer.writerow(row)

def get_common_time(mpc, campc):
    N = min(mpc['time'].size, campc['time'].size)
    return mpc['time'][:N]

def tracking_error(ref, sim):
    rstate = ref['state']
    sstate = sim['state']
    N = min(sstate.shape[1], rstate.shape[1])
    tracking_err = (rstate[:, :N]  - sstate[:, :N])**2
    final_err = np.abs(sstate[:, -1] - rstate[:, -1])
    return tracking_err, final_err

def feedback_effort(ref, sim):
    rcontrol = ref['control']
    scontrol = sim['control']
    N = min(rcontrol.shape[1], scontrol.shape[1])
    effort = (rcontrol[:,:N] - scontrol[:,:N])**2
    return effort

def get_forces_from_logs(logs):
    force_vars = np.column_stack([log['force'][:, 0] for log in logs])
    return FORCE_MATRIX.dot(force_vars)    

def do_tracking_error_analysis():
    print('Running tracking error analysis')
    refdata = utils.load(REFDATA)
    fig, axs = plt.subplots(4,1)
    mpc_terr = ['MPC']
    cpc_terr = ['CAMPC']
    mpc_ferr = ['MPC']
    cpc_ferr = ['CAMPC']

    for k, part in enumerate(PARTS):
        mpcdata = utils.load(os.path.join(SOURCE, part, MPC_SIM))
        cpcdata = utils.load(os.path.join(SOURCE, part, CPC_SIM))    
        mpc_track, mpc_final = tracking_error(refdata, mpcdata)
        cpc_track, cpc_final = tracking_error(refdata, cpcdata)
        mpc_terr.append(np.sqrt(np.mean(mpc_track[0,:])))
        cpc_terr.append(np.sqrt(np.mean(cpc_track[0,:])))
        mpc_ferr.append(mpc_final[0])
        cpc_ferr.append(cpc_final[0])
        t = get_common_time(refdata, mpcdata)
        axs[k].plot(t, mpc_track[0,:], linewidth=1.5, label='MPC')
        axs[k].plot(t, cpc_track[0,:], linewidth=1.5, label='CAMPC')
        axs[k].set_ylabel(part)
    axs[0].set_title('Position Tracking Error (m^2)')
    axs[-1].set_xlabel('Time (s)')
    axs[0].legend(frameon = False)
    fig.savefig(os.path.join(SOURCE, 'TrackingError.pdf'), dpi=fig.dpi, bbox_inches='tight')
    print("\tSaved figure!")
    # Write data to csv file
    colnames = ['']
    colnames.extend(PARTS)
    write_to_file(os.path.join(SOURCE, 'tracking_error.csv'), [colnames, mpc_terr, cpc_terr])
    write_to_file(os.path.join(SOURCE, 'final_error.csv'), [colnames, mpc_ferr, cpc_ferr])
    print("\tSaved csv data")

def do_feedback_effort_analysis():
    print('Running feedback effort analysis')
    fig, axs = plt.subplots(4, 1)
    refdata = utils.load(REFDATA)
    m_effort = ['MPC']
    c_effort = ['CAMPC']

    for k, part in enumerate(PARTS):
        mpcdata = utils.load(os.path.join(SOURCE, part, MPC_SIM))
        cpcdata = utils.load(os.path.join(SOURCE, part, CPC_SIM))
        mpc_effort = feedback_effort(refdata, mpcdata)
        cpc_effort = feedback_effort(refdata, cpcdata)
        m_effort.append(np.sqrt(np.mean(mpc_effort[0,:])))
        c_effort.append(np.sqrt(np.mean(cpc_effort[0,:])))
        t = get_common_time(refdata, mpcdata)
        axs[k].plot(t, mpc_effort[0,:], linewidth=1.5, label='MPC')
        axs[k].plot(t, cpc_effort[0,:], linewidth=1.5, label='CAMPC')
        axs[k].set_ylabel(part)
    axs[0].set_title('Feedback Effort (N^2)')
    axs[-1].set_xlabel('Time (s)')
    axs[0].legend(frameon=False)
    fig.savefig(os.path.join(SOURCE, 'FeedbackEffort.pdf'), dpi=fig.dpi, bbox_inches='tight')
    print("\tSaved figure")
    # Write data to csv file
    colnames = ['']
    colnames.extend(PARTS)
    write_to_file(os.path.join(SOURCE, 'feedback_effort.csv'), [colnames, m_effort, c_effort])
    print('\tSaved CSV')

def do_predicted_force_analysis():
    print('Running predicted force analysis')
    fig, axs = plt.subplots(4,1)
    mpc_force_total = ['MPC']
    cpc_force_total = ['CAMPC']
    for k, part in enumerate(PARTS):
        mpc_logs = OptimizationLogger.load(os.path.join(SOURCE, part, MPC_LOG)).logs
        cpc_logs = OptimizationLogger.load(os.path.join(SOURCE, part, CPC_LOG)).logs
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
        axs2[-1].set_xlabel('Time (s)')
        axs2[0].set_title(f'{part} MPC Force Mismatch')
        axs3[-1].set_xlabel('Time (s)')
        axs3[0].set_title(f'{part} CAMPC Force Mismatch')
        axs2[0].legend(frameon=False)
        axs3[0].legend(frameon=False)
        fig2.savefig(os.path.join(SOURCE, part + '_MPC_Forces.pdf'), dpi=fig2.dpi, bbox_inches='tight')
        fig3.savefig(os.path.join(SOURCE, part + '_CAMCP_Forces.pdf'), dpi=fig3.dpi, bbox_inches='tight')
        print(f"\tSaved figures for {part}")
        # plot the overall force mismatch
        mpc_err = np.sqrt(np.mean((mpc_force - mpc_pred_force)**2, axis=0))
        cpc_err = np.sqrt(np.mean((cpc_force - cpc_pred_force)**2, axis=0))
        t = get_common_time(mpcdata, cpcdata)
        axs[k].plot(t[1:], mpc_err, linewidth=1.5, label='MPC')
        axs[k].plot(t[1:], cpc_err, linewidth=1.5, label='CAMPC')
        axs[k].set_ylabel(part)
        # Record the total force error
        mpc_force_total.append(np.mean(mpc_err))
        cpc_force_total.append(np.mean(cpc_err))

    axs[-1].set_xlabel('Time (s)')
    axs[0].set_title('Force Mismatch (N)')  
    axs[0].legend(frameon=False)  
    fig.savefig(os.path.join(SOURCE, 'ForceMismatch.pdf'), dpi=fig.dpi, bbox_inches='tight')
    print(f"\tSaved common figure")
    # Write data to csvfile
    colnames = ['']
    colnames.extend(PARTS)
    write_to_file(os.path.join(SOURCE, 'predicted_force_error.csv'), [colnames, mpc_force_total, cpc_force_total])
    print(f"\tSaved CSV")

if __name__ == '__main__':
    #do_feedback_effort_analysis()
    #do_tracking_error_analysis()
    do_predicted_force_analysis()