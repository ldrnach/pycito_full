import os, errno
import utilities as utils
import numpy as np
from trajopt.optimizer import A1OptimizerConfiguration, A1VirtualBaseOptimizer
from pydrake.all import PiecewisePolynomial
import matplotlib.pyplot as plt

def load_configuration_data(source, config):
    filename = os.path.join(source, config)
    if os.path.isfile(filename):
        return A1OptimizerConfiguration.load(filename)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

def load_trajectory_data(source, trajectory):
    return utils.load(os.path.join(source, trajectory))

def linear_resample_trajectory_data(trajdata, num_samples):
    keys = ["state","control","force","jointlimit","slacks"]
    new_time = np.linspace(trajdata['time'][0], trajdata['time'][-1], num_samples)
    for key in keys:
        if key in trajdata.keys() and trajdata[key] is not None:
            traj = PiecewisePolynomial.FirstOrderHold(trajdata['time'], trajdata[key])
            trajdata[key] = traj.vector_values(new_time)
    return trajdata

def upsample_and_warmstart(optimizer, trajdata):
    newdata = linear_resample_trajectory_data(trajdata, optimizer.trajopt.num_time_samples)
    optimizer.useCustomGuess(x_init=newdata['state'], u_init=newdata['control'], l_init=newdata['force'], jl_init=newdata['jointlimit'])
    if newdata['slacks'] is not None:
        optimizer.trajopt.set_initial_guess(straj=newdata['slacks'])
    return optimizer


def run_a1_mesh_refinement(source, config, trajectory, num_samples_new, outdir):
    # Load the configuration
    configdata = load_configuration_data(source, config)
    # Change the sampling 
    configdata.num_time_samples = num_samples_new
    # Load trajectory data
    trajdata   = load_trajectory_data(source, trajectory)
    # Create the optimization
    optimizer = A1VirtualBaseOptimizer.buildFromConfig(configdata)
    # Warmstart the optimization
    optimizer = upsample_and_warmstart(optimizer, trajdata)
    # Run the optimization, save the results
    results = optimizer.solve()
    # Save the results
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    optimizer.saveResults(results, name=os.path.join(outdir, 'trajoptresults.pkl'))
    optimizer.saveReport(results, savename=os.path.join(outdir, 'report.txt'))
    optimizer.saveDebugFigure(savename=os.path.join(outdir, 'CostsAndConstraints.png'))
    optimizer.plot(results, show=False, savename=os.path.join(outdir, 'trajopt.png'))
    plt.close('all')

if __name__ == "__main__":
    source = os.path.join("examples","a1","runs","Jul-07-2021","lifting_N51_VelCost50_ComplCost10equaltime")
    config = 'lifting_N51_VelCost50_ComplCost10equaltime.pkl'
    trajectory = 'trajoptresults.pkl'
    outdir = os.path.join('examples','a1','lifting_upsamples_N101')
    num_samples_new = 101
    run_a1_mesh_refinement(source, config, trajectory, num_samples_new, outdir)