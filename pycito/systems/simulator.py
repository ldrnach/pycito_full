"""
Contact Simulation 

Luke Drnach
February 8, 2022
"""
#TODO: THRUSDAY: Unit test all simulators

import numpy as np

class Simulator():
    def __init__(self, plant):
        self.plant = plant
        self._timestep = 0.01

    @staticmethod
    def Uncontrolled(plant):
        return Simulator(plant)

    @staticmethod
    def OpenLoop(plant, utraj):
        return OpenLoopSimulator(plant, utraj)

    @staticmethod
    def ClosedLoop(plant, controller):
        return ClosedLoopSimulator(plant, controller)

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
            control[:, n] = self.get_control(time[n-1], state[:, n-1])
            force[:, n] = self.plant.contact_impulse(self._timestep, state[:, n-1], control[:, n])
            state[:, n] = self.plant.integrate(self._timestep, state[:, n-1], control[:, n], force[:, n])            
        # Return the simulation values
        return time, state, control, force

    def get_control(self, t, x):
        """Uncontrolled simulator returns the null control"""
        return np.zeros((self.plant.multibody.num_actuators(), ))

    @property
    def timestep(self):
        return self._timestep

    @timestep.setter
    def timestep(self, val):
        if isinstance(val, [int, float]) and val > 0:
            self._timestep = val
        else:
            raise ValueError(f"timestep must be a nonnegative number")

class OpenLoopSimulator(Simulator):
    def __init__(self, plant, utraj):
        super(OpenLoopSimulator, self).__init__(plant)
        self.utraj = utraj

    def get_control(self, t, x):
        """Get the open loop control value"""
        if t < self.utraj.start_time():
            return self.utraj.value(self.utraj.start_time())
        elif t > self.utraj.end_time():
            return self.utraj.value(self.utraj.end_time())
        else:
            return self.utraj.value(t)

class ClosedLoopSimulator(Simulator):
    def __init__(self, plant, controller):
        super(ClosedLoopSimulator, self).__init__(plant)
        self.controller = controller

    def get_control(self, t, x):
        """Get the closed loop (MPC) control value"""
        return self.controller.do_mpc(t, x)
    
class ContactEstimationSimulator(Simulator):
    pass

if __name__ == '__main__':
    print("Hello from simulator!")