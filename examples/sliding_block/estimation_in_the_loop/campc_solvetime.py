"""
Runs analysis for the campc simulations

"""
import os, csv
import numpy as np
from pycito.controller.optimization import OptimizationLogger

SOURCE = os.path.join('examples','sliding_block','estimation_in_the_loop','final_for_paper')
PARTS = ['flatterrain','stepterrain','high_friction','low_friction']

CPC_LOG = os.path.join('campc_logs','mpclogs.pkl')
MPC_LOG = os.path.join('mpc_logs','mpclogs.pkl')
EST_LOG = os.path.join('campc_logs','EstimationLogs.pkl')

def write_to_file(filename, rowdata):
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        for row in rowdata:
            writer.writerow(row)

def get_solvetime_statistics(logdata):
    times = np.array([log['solvetime'] for log in logdata])
    return np.mean(times), np.std(times)

def do_solvetime_analysis():
    print('Running solvetime analysis')

    mean_mpc_time = ['MPC'] + [0] * 4
    std_mpc_time = ['MPC'] + [0] * 4
    mean_cpc_time = ['CAMPC'] + [0] * 4
    std_cpc_time = ['CAMPC'] + [0] * 4
    mean_est_time = ['Estimation'] + [0] * 4
    std_est_time = ['Estimation'] + [0] * 4

    for k, part in enumerate(PARTS):
        mpclogs = OptimizationLogger.load(os.path.join(SOURCE, part, MPC_LOG)).logs
        cpclogs = OptimizationLogger.load(os.path.join(SOURCE, part, CPC_LOG)).logs
        estlogs = OptimizationLogger.load(os.path.join(SOURCE, part, EST_LOG)).logs
        mean_mpc_time[k+1], std_mpc_time[k+1] = get_solvetime_statistics(mpclogs)
        mean_cpc_time[k+1], std_cpc_time[k+1] = get_solvetime_statistics(cpclogs)
        mean_est_time[k+1], std_est_time[k+1] = get_solvetime_statistics(estlogs)
    # Write data to csvfile
    colnames=['']
    colnames.extend(PARTS)
    write_to_file(os.path.join(SOURCE, 'mean_solvetimes.csv'), [colnames, mean_mpc_time, mean_cpc_time, mean_est_time])
    write_to_file(os.path.join(SOURCE, 'std_solvetimes.csv'), [colnames, std_mpc_time, std_cpc_time, std_est_time])
    print(f"\tSaved CSV")

if __name__ == '__main__':
    do_solvetime_analysis()