import os
import numpy as np
import matplotlib.pyplot as plt

from pycito.controller.optimization import OptimizationLogger
import pycito.utilities as utils

SOURCE = os.path.join('data','slidingblock','speedtests')
GROUPS = ['flatterrain','stepterrain','highfriction']
MPCPATH = os.path.join('campc_logs','mpclogs.pkl')
ESTPATH = os.path.join('campc_logs','EstimationLogs.pkl')

REFPATH = os.path.join('data','slidingblock','block_reference.pkl')
CPCSIM = os.path.join('campcsim.pkl')
MPCSIM = os.path.join('mpcsim.pkl')

HORIZONS = range(1, 21)

FIG_EXT = '.pdf'

def tracking_error(ref, sim):
    rstate = ref['state']
    sstate = sim['state']
    N = min(sstate.shape[1], rstate.shape[1])
    tracking_err = np.sqrt(np.mean((rstate[:, :N]  - sstate[:, :N])**2, axis=1))
    final_err = np.abs(sstate[:, -1] - rstate[:, -1])
    return tracking_err[0], final_err[0]
    
def feedback_effort(ref, sim):
    rcontrol = ref['control']
    scontrol = sim['control']
    N = min(rcontrol.shape[1], scontrol.shape[1])
    effort = np.sqrt(np.mean((rcontrol[:,:N] - scontrol[:,:N])**2))
    return effort

# Make timing performance plots
mpc_means = np.zeros((3, 20))
mpc_stdev = np.zeros((3, 20))
est_means = np.zeros((3, 20))
est_stdev = np.zeros((3, 20))
mpc_solve = np.zeros((3, 20))
est_solve = np.zeros((3, 20))

fig, axs = plt.subplots(2,1)
fig2, axs2 = plt.subplots(2,1)

for n, group in enumerate(GROUPS):
    for k, horizon in enumerate(HORIZONS):
        mpclogs = OptimizationLogger.load(os.path.join(SOURCE, group, f'horizon_{horizon}', MPCPATH)).logs
        estlogs = OptimizationLogger.load(os.path.join(SOURCE, group, f'horizon_{horizon}', ESTPATH)).logs
        mpc_time = np.array([log['solvetime'] for log in mpclogs])
        est_time = np.array([log['solvetime'] for log in estlogs]) 
        mpc_success = [log['success'] for log in mpclogs]
        est_success = [log['success'] for log in estlogs]       
        # Store the data
        mpc_solve[n, k] = sum(mpc_success)/len(mpc_success)
        est_solve[n, k] = sum(est_success)/len(mpc_success)
        mpc_means[n, k], mpc_stdev[n, k] = np.mean(mpc_time[mpc_success]), np.std(mpc_time[mpc_success])/np.sqrt(sum(mpc_success))
        est_means[n, k], est_stdev[n, k] = np.mean(est_time[est_success]), np.std(est_time[est_success])/np.sqrt(sum(est_success))

    axs[0].plot(HORIZONS, mpc_means[n, :], linewidth=1.5, label=group)
    axs[0].fill_between(HORIZONS, mpc_means[n, :]-mpc_stdev[n, :], mpc_means[n, :] + mpc_stdev[n, :], alpha=0.5)
    axs[1].plot(HORIZONS, mpc_solve[n, :], linewidth=1.5)
    
    axs2[0].plot(HORIZONS, est_means[n,:], linewidth=1.5, label=group)
    axs2[0].fill_between(HORIZONS, est_means[n,:] - est_stdev[n,:], est_means[n,:] + est_stdev[n,:], alpha=0.5)
    axs2[1].plot(HORIZONS, est_solve[n, :], linewidth=1.5)

axs[0].set_title('MPC Performance')
axs[0].set_ylabel('Solve Time (s)')
axs[0].set_yscale('log')
axs[0].grid()
axs[1].set_ylabel('Solve Success Rate')
axs[1].set_xlabel('Horizon (N)')
axs[0].legend(frameon=False)

axs2[0].set_title('Estimator Performance')
axs2[0].set_ylabel('Solve Time (s)')
axs2[0].set_yscale('log')
axs2[0].grid()
axs2[1].set_ylabel('Solve Success Rate')
axs2[1].set_xlabel('Horizon (N)')
axs2[0].legend(frameon=False)

fig.savefig(os.path.join(SOURCE, 'MPC_Performance' + FIG_EXT), dpi=fig.dpi, bbox_inches='tight')
fig2.savefig(os.path.join(SOURCE, 'Estimator_Performance' + FIG_EXT), dpi=fig2.dpi, bbox_inches='tight')


# Make tracking performance plots
reftraj = utils.load(REFPATH)
fig1, axs1 = plt.subplots(3,1)
fig2, axs2 = plt.subplots(3,1)
fig3, axs3 = plt.subplots(3,1)
mpc_tracking = np.zeros((3,20))
cpc_tracking = np.zeros((3,20))
mpc_final = np.zeros((3, 20))
cpc_final = np.zeros((3, 20))
mpc_effort = np.zeros((3, 20))
cpc_effort = np.zeros((3, 20))

mpc_tracking[:] = np.nan
cpc_tracking[:] = np.nan
mpc_final[:] = np.nan
cpc_final[:] = np.nan
mpc_effort[:] = np.nan
cpc_effort[:] = np.nan



for n, group in enumerate(GROUPS):
    for k, horizon in enumerate(HORIZONS):
        mpcsim = utils.load(os.path.join(SOURCE, group, f'horizon_{horizon}', MPCSIM))
        cpcsim = utils.load(os.path.join(SOURCE, group, f'horizon_{horizon}', CPCSIM))
        if mpcsim['status']:
            mpc_tracking[n, k], mpc_final[n, k] = tracking_error(reftraj, mpcsim)
            mpc_effort[n,k] = feedback_effort(reftraj, mpcsim)
        if cpcsim['status']:
            cpc_tracking[n, k], cpc_final[n, k] = tracking_error(reftraj, cpcsim)
            cpc_effort[n, k] = feedback_effort(reftraj, cpcsim)

    axs1[n].plot(HORIZONS, mpc_tracking[n,:], linewidth=1.5, label='MPC')
    axs1[n].plot(HORIZONS, cpc_tracking[n,:], linewidth=1.5, label='CAMPC')
    axs1[n].set_ylabel(f"{group}\n Error (m)")
    axs1[n].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    axs2[n].plot(HORIZONS, mpc_final[n,:], linewidth=1.5, label='MPC')
    axs2[n].plot(HORIZONS, cpc_final[n,:], linewidth=1.5, label='CAMPC')
    axs2[n].set_ylabel(f"{group}\n Error (m)")
    axs2[n].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    axs3[n].plot(HORIZONS, mpc_effort[n,:], linewidth=1.5, label='MPC')
    axs3[n].plot(HORIZONS, cpc_effort[n,:], linewidth=1.5, label='CAMPC')
    axs3[n].set_ylabel(f"{group}\n Effort (N)")
    axs3[n].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

axs3[0].set_ylim(0,5)
axs3[2].set_ylim(0,10)
axs1[0].set_title('Position Tracking Error')
axs1[2].set_xlabel('Horizon (N)')
axs1[0].legend(frameon=False)
fig1.tight_layout()

axs2[0].set_title('Final Position Error')
axs2[2].set_xlabel('Horizon (N)')
axs2[0].legend(frameon=False)
fig2.tight_layout()

axs3[0].set_title('Feedback Control Effort')
axs3[2].set_xlabel('Horizon (N)')
axs3[0].legend(frameon=False)
fig3.tight_layout()

fig1.savefig(os.path.join(SOURCE, 'TrackingError' + FIG_EXT), dpi=fig1.dpi, bbox_inches='tight')
fig2.savefig(os.path.join(SOURCE, 'FinalPositionError' + FIG_EXT), dpi=fig2.dpi, bbox_inches='tight')
fig3.savefig(os.path.join(SOURCE, 'FeedbackEffort' + FIG_EXT), dpi=fig3.dpi, bbox_inches='tight')
print('Figures Saved!')