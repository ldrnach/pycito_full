import os
from pycito.systems.A1.a1 import A1VirtualBase
import a1_mpc_tools as mpctools
from pycito.utilities import save, load

DISTANCE = 3
SOURCE = os.path.join('data','a1','reference',f'{DISTANCE}m','reftraj.pkl')
TARGET = os.path.join('examples','a1','estimation_in_the_loop','mpc','flatterrain_relaxed',f'{DISTANCE}m','test')

def make_flatterrain_model():
    a1 = A1VirtualBase()
    a1.terrain.friction = 1.0
    a1.Finalize()
    return a1

def main():
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)
    # Make the simuation
    model = make_flatterrain_model()
    reftraj = mpctools.get_reference_trajectory(SOURCE)
    controller = mpctools.make_mpc_controller(reftraj)
    # controller.enable_cost_display(display='terminal')
    # controller.enable_cost_display(display='terminal')
    # Run the simulation
    simdata = mpctools.run_simulation(model, controller, duration=1.)
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

if __name__ == '__main__':
    main()