import os
import numpy as np
from controllers import A1StandingPDController
from environments import FlatGroundEnvironment
from a1_simulator import A1DrakeSimulationBuilder, A1SimulationPlotter

from analysistools import plot_tracking_error

from pycito.controller.mpc import ReferenceTrajectory

savedir = os.path.join('drake_simulation','mpc_standing_sim')
if not os.path.exists(savedir):
    os.makedirs(savedir)

class A1StandingReference():
    def __init__(self, t, state):
        self._time = t
        self._state = state

# Simulation specification
timestep = 0.01
sim_time = 2 #Seconds
env = FlatGroundEnvironment()
control  = A1StandingPDController

# Build and run the simulation
simulation, simbuilder = A1DrakeSimulationBuilder.createSimulator(timestep, env, control)

# Set the initial condition
initial_state = simbuilder.get_a1_standing_state()
simbuilder.set_initial_state(initial_state)

# Run the simulation
simbuilder.initialize_sim()
simbuilder.run_simulation(sim_time)
print('Simulation complete')

# Get and plot the logs from the simulation
contactdict = simbuilder.contact_logger.to_dictionary()
simplotter = A1SimulationPlotter()
simplotter.plot(simbuilder.get_logs(), show=False, savename=os.path.join(savedir, 'simdata.png'))
# Plot the tracking errors
simdata = simbuilder.get_simulation_data()
ref = np.concatenate([simbuilder.controller.q_ref, simbuilder.controller.v_ref], axis=0)
reftraj = A1StandingReference([0, sim_time], state = np.column_stack([ref, ref]))

plot_tracking_error(simdata, reftraj, savedir)
