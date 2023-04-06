import os

from drake_simulation.controllers import A1ContactMPCController
from drake_simulation.environments import RampUpEnvironment
from drake_simulation.a1_simulator import A1DrakeSimulationBuilder, A1SimulationPlotter
import drake_simulation.analysistools as analysistools

from pycito.systems.A1.a1 import A1

target = os.path.join('drake_simulation','killer_experiment','rampup')
if not os.path.exists(target):
    os.makedirs(target)

# Simulation specification
timestep = 0.01
sim_time = 12.0 #Seconds
env = RampUpEnvironment()
control  = A1ContactMPCController

# Build and run the simulation
simulation, simbuilder = A1DrakeSimulationBuilder.createSimulator(timestep, env, control)
#simbuilder.plant.set_penetration_allowance(0.001)
# Set the initial condition
initial_state = simbuilder.get_a1_standing_state()
simbuilder.set_initial_state(initial_state)
simbuilder.controller.enable_logging()
# Run the simulation
simbuilder.initialize_sim()
simbuilder.run_simulation(sim_time)
print('Simulation complete')

# Get and plot the logs from the simulation
simplotter = A1SimulationPlotter()
simplotter.plot(simbuilder.get_logs(), show=False, savename=os.path.join(target, 'sim.png'))
simplotter.save_data(simbuilder.get_logs(), savename=os.path.join(target, 'simdata.pkl'))

data = simbuilder.get_simulation_data()
ref = simbuilder.controller.get_reference_trajectory()

analysistools.plot_tracking_error(data, ref, target)
#plot_tracking_error(data, ref, savedir=os.path.join(target))
analysistools.save_mpc_logs(simbuilder.controller.get_mpc(), target)
analysistools.plot_mpc_logs(simbuilder.controller.get_mpc(), target)