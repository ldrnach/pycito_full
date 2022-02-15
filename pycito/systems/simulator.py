"""
Contact Simulation 

Luke Drnach
February 8, 2022
"""

import numpy as np
import pycito.controller.mpc as mpc

class Simulator():
    def __init__(self, plant, controller):
        self.plant = plant
        self.controller = controller
        self._timestep = 0.01

    @staticmethod
    def Uncontrolled(plant):
        return Simulator(plant, mpc.NullController(plant))

    def simulate(self, initial_state, duration):
        """Run the simulation"""
        # Initialize the output arrays
        N = int(duration/self._timestep)
        while (N-1)*self._timestep < duration:
            N += 1
        time = np.zeros((N,))
        time[1:] = self._timestep
        time = np.cumsum(time)
        # State
        state = np.zeros((self.plant.multibody.num_positions() + self.plant.multibody.num_velocities(), N))
        state[:, 0] = initial_state
        # Everything else
        control = np.zeros((self.plant.multibody.num_actuators(), N))
        force  = np.zeros((self.plant.num_contacts() + self.plant.num_friction(), N))
        # Run the simulation
        for n in range(1, N):
            control[:, n] = self.controller.get_control(time[n-1], state[:, n-1])
            force[:, n] = self.plant.contact_impulse(self._timestep, state[:, n-1], control[:, n])
            state[:, n] = self.plant.integrate(self._timestep, state[:, n-1], control[:, n], force[:, n])            
        # Return the simulation values
        return time, state, control, force

    @property
    def timestep(self):
        return self._timestep

    @timestep.setter
    def timestep(self, val):
        if isinstance(val, [int, float]) and val > 0:
            self._timestep = val
        else:
            raise ValueError(f"timestep must be a nonnegative number")

if __name__ == '__main__':
    print("Hello from simulator!")