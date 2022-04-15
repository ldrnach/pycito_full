"""
Contact Simulation 

Luke Drnach
February 8, 2022
"""
#TODO: Determine a good method for integrating the contact dynamics
import numpy as np
import pycito.controller.mpc as mpc
from pycito.systems.integrators import ContactDynamicsIntegrator
import pycito.utilities as utils

class Simulator():
    def __init__(self, plant, controller):
        self.plant = plant
        self.controller = controller
        self._timestep = np.array([0.01])
        self.integrator = ContactDynamicsIntegrator.ImplicitMidpointIntegrator(plant)

    @staticmethod
    def Uncontrolled(plant):
        return Simulator(plant, mpc.NullController(plant))

    def useImplicitEuler(self):
        self.integrator = ContactDynamicsIntegrator.ImplicitEulerIntegrator(self.plant)

    def useSemiImplicitEuler(self):
        self.integrator = ContactDynamicsIntegrator.SemiImplicitEulerIntegrator(self.plant)

    def useImplicitMidpoint(self):
        self.integrator = ContactDynamicsIntegrator.ImplicitMidpointIntegrator(self.plant)

    def useTimestepping(self):
        self.integrator = self.plant

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
        control = np.zeros((self.integrator.u.shape[0], N))
        force  = np.zeros((self.integrator.forces.shape[0], N))
        # Run the simulation
        status = True
        n = 1
        while n < N and status:
            control[:, n-1] = self.controller.get_control(time[n-1], state[:, n-1]) #FIX THIS. THE INDEX IS WRONG
            state[:, n], force[:, n], status = self.integrator.integrate(self._timestep, state[:, n-1], control[:, n-1])     
            n += 1
        # Return the simulation values
        if status:
            return time, state, control, force, status
        else:
            return time[:n-1], state[:, n-1], control[:, n-1], force[:, n-1], status
        
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