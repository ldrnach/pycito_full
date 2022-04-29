"""
Contact Simulation 

Luke Drnach
February 8, 2022
"""
#TODO: Determine a good method for integrating the contact dynamics
import numpy as np
import pycito.controller.mpc as mpc
from pycito.trajopt import complementarity as cp
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

    def useImplicitEuler(self, ncp=cp.NonlinearConstantSlackComplementarity):
        self.integrator = ContactDynamicsIntegrator.ImplicitEulerIntegrator(self.plant, ncp)

    def useSemiImplicitEuler(self, ncp=cp.NonlinearConstantSlackComplementarity):
        self.integrator = ContactDynamicsIntegrator.SemiImplicitEulerIntegrator(self.plant, ncp)

    def useImplicitMidpoint(self, ncp=cp.NonlinearConstantSlackComplementarity):
        self.integrator = ContactDynamicsIntegrator.ImplicitMidpointIntegrator(self.plant, ncp)

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
        control = np.zeros((self.plant.multibody.num_actuators(), N))
        force  = np.zeros((self.plant.num_contacts() + self.plant.num_friction(), N))
        # Get the initial static control law
        control[:, 0], fN = self.plant.static_controller(state[:self.plant.multibody.num_positions(), 0])
        force[:self.plant.num_contacts(), 0] = fN
        # Run the simulation
        status = True
        for n in range(1, N):
            control[:, n] = self.controller.get_control(time[n-1], state[:, n-1], control[:, n-1])
            force[:, n] = self.plant.contact_impulse(self._timestep, state[:, n-1], control[:, n])
            state[:, n] = self.plant.integrate(self._timestep, state[:, n-1], control[:, n], force[:, n])
            force[:, n] = force[:, n] / self._timestep
            if np.any(np.isnan(state[:, n])):
                control = control[:, :n-1]
                force = force[:, :n-1] 
                state = state[:, :n-1]
                time = time[:n-1]
                status = False
                break            
        # Return the simulation values
        if status:
            return time, state, control, force, status
        else:
            return time[:n-1], state[:, :n-1], control[:, :n-1], force[:, :n-1], status
        
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