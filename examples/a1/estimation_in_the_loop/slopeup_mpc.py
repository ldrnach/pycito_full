import os
import numpy as np
import pycito.systems.contactmodel as cm
from pycito.systems.A1.a1 import A1VirtualBase
import a1_mpc_tools as mpctools
from pycito.utilities import save, load

DISTANCE = 3
SOURCE = os.path.join('data','a1','reference',f'{DISTANCE}m','reftraj.pkl')
TARGET = os.path.join('examples','a1','estimation_in_the_loop','mpc','slopeup',f'{DISTANCE}m')

SLOPE_START = (1, 0)
SLOPE_STOP = (2, 0.35) # These parameters give a slope of about 20 degrees for 1m

def make_slopeup_terrain():
    # Calculate slope parameters   
    slope = (SLOPE_STOP[1] - SLOPE_START[1])/(SLOPE_STOP[0] - SLOPE_START[0])
    intercept = SLOPE_START[1] - slope * SLOPE_START[0]
    slope_vec = np.array([-slope, 0, 1])
    # Create terrain
    breaks = [SLOPE_START[0], SLOPE_STOP[0]]
    models = [
        cm.FlatModel(location = 0, direction = np.array([0, 0, 1])),
        cm.FlatModel(location = intercept, direction = slope_vec),
        cm.FlatModel(location = SLOPE_STOP[1], direction = np.array([0, 0, 1]))
    ]
    return cm.ContactModel(surface = cm.PiecewiseModel(breaks, models), friction = cm.ConstantModel(const = 1.))
    

def make_slopeup_model():
    a1 = A1VirtualBase(terrain = make_slopeup_terrain())
    a1.Finalize()
    return a1


def main():
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)
    # Make the simuation
    model = make_slopeup_model()
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

def plot_slope_terrain():
    terrain = make_slopeup_terrain()
    x = np.linspace(np.array([0, 0, 0]), np.array([3, 0, 0]), 100).T
    terrain.plot2D(x, show=True)

if __name__ == '__main__':
    main()