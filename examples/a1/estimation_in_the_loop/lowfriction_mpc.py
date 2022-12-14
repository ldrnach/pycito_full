import os
import numpy as np
import pycito.systems.terrain as terrain
from pycito.systems.A1.a1 import A1VirtualBase
import a1_mpc_tools as mpctools
from pycito.utilities import save, load

DISTANCE = 3
SOURCE = os.path.join('data','a1','reference',f'{DISTANCE}m','reftraj.pkl')
TARGET = os.path.join('examples','a1','estimation_in_the_loop','mpc','lowfriction',f'{DISTANCE}m')

def low_friction(x):
    if x[0] > 1.0 and x[0] < 2.0:
        return 0.5
    else:
        return 1.0

def make_lowfriction_model():
    lowfric = terrain.VariableFrictionFlatTerrain(height = 0, fric_func = low_friction)
    a1 = A1VirtualBase(terrain = lowfric)
    a1.Finalize()
    return a1

def main():
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)
    # Make the simuation
    model = make_lowfriction_model()
    reftraj = mpctools.get_reference_trajectory(SOURCE)
    controller = mpctools.make_mpc_controller(reftraj)
    # Run the simulation
    simdata = mpctools.run_simulation(model, controller)
    # Save and plot the results
    save(os.path.join(TARGET, 'simdata.pkl'), simdata)
    mpctools.plot_mpc_logs(controller, TARGET)
    mpctools.save_mpc_logs(controller, TARGET)
    mpctools.plot_sim_results(model, simdata, TARGET)
    mpctools.plot_tracking_error(simdata, reftraj, TARGET)

def tracking_error():
    simdata = load(os.path.join(TARGET, 'simdata.pkl'))
    reftraj = mpctools.get_reference_trajectory(SOURCE)
    mpctools.plot_tracking_error(simdata, reftraj, TARGET)

def plot_terrain():
    terrain = make_lowfriction_model().terrain
    x = np.linspace(np.array([0, 0, 0]), np.array([3, 0, 0]), 100).T
    terrain.plot2D(x, show=True)

if __name__ == '__main__':
    main()