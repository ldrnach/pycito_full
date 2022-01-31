"""
Class for Linear Contact-Implicit MPC

January 26, 2022
Luke Drnach
"""
#TODO: Make a class for linearizing constraints around a trajectory
#TODO: Write class for running MPC on linearized constraints
#TODO: Thought - have LinearizedContactTrajectory store the MLCP constraints instead of the parameters

import pycito.utilities as utils
import pycito.trajopt.constraints as cstr

class LinearizedContactTrajectory():
    def __init__(self, plant, xtraj, utraj, ftraj):
        self.plant = plant
        # Store the trajectory values
        self.time, self.state = utils.GetKnotsFromTrajectory(xtraj)
        self.control = utraj.vector_values(self.time)

        # Store the linearized parameter values
        self._linearize_dynamics()
        self._linearize_normal_distance()
        self._linearize_maximum_dissipation()
        self._linearize_friction_cone()


    def _linearize_dynamics(self):
        pass


    def _linearize_normal_distance(self):
        """Store the linearizations for the normal distance constraint"""
        distance = cstr.NormalDistanceConstraint(self.plant)
        self.distance_params = [distance.linearize(x) for x in self.state.transpose()]

    def _linearize_maximum_dissipation(self):
        """Store the linearizations for the maximum dissipation function"""
        dissipation = cstr.MaximumDissipationConstraint(self.plant)
        self.dissipation_params = [dissipation.linearize(x, s) for x, s in zip(self.state.transpose(), self.slacks.transpose())]

    def _linearize_friction_cone(self):
        """Store the linearizations for the friction cone constraint function"""
        friccone = cstr.FrictionConeConstraint(self.plant)
        self.friccone_params = [friccone.linearize(x, f) for x, f in zip(self.state.transpose(), self.forces.transpose())]


class LinearContactMPC():
    def __init__(self, plant, traj):
        """
        Plant: a TimesteppingMultibodyPlant instance
        Traj: a LinearizedContactTrajectory
        """
        pass

if __name__ == "__main__":
    print("Hello from MPC!")