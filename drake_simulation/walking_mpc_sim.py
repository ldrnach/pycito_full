import os

from controllers import A1ContactMPCController
from environments import FlatGroundEnvironment
from a1_simulator import A1DrakeSimulationBuilder, A1SimulationPlotter

from pycito.systems.A1.a1 import A1

target = os.path.join('drake_simulation','mpc_walking_sim')
if not os.path.exists(target):
    os.makedirs(target)

# Simulation specification
timestep = 0.01
sim_time = 2 #Seconds
env = FlatGroundEnvironment()
control  = A1ContactMPCController

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
#simplotter = A1SimulationPlotter()
#simplotter.plot(simbuilder.get_logs(), show=False, savename=os.path.join(target, 'sim.png'))
#simplotter.save_data(simbuilder.get_logs(), savename=os.path.join(target, 'simdata.pkl'))