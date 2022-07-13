from controllers import A1StandingPDController
from environments import FlatGroundEnvironment
from a1_simulator import A1DrakeSimulationBuilder, A1SimulationPlotter

from pycito.systems.A1.a1 import A1

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
A1SimulationPlotter().plot(simbuilder.get_logs())