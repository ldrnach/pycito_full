import os
import numpy as np

import pycito.utilities as utils
from pycito.systems.A1.a1 import A1

from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.geometry import  MeshcatVisualizerParams, Role, StartMeshcat
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.multibody.parsing import Parser

from pydrake.all import MeshcatVisualizer, SceneGraph, MultibodyPlant, LogVectorOutput, PiecewisePolynomial

URDF = os.path.join("systems","A1","A1_description","urdf","a1_foot_collision.urdf")

#TODO: Refactor. Use https://github.com/RobotLocomotion/drake/blob/master/tutorials/authoring_multibody_simulation.ipynb as a reference
#TODO: Finish adding meshcat visualization

class A1DrakeSimulationBuilder():
    def __init__(self, timestep = 0.01):
        path = utils.FindResource(URDF)
        # Setup the diagram
        self.timestep = timestep
        self.builder = DiagramBuilder()
        self.scene_graph = self.builder.AddSystem(SceneGraph())
        self.plant = self.builder.AddSystem(MultibodyPlant(time_step = timestep))
        self.plant.RegisterAsSourceForSceneGraph(self.scene_graph)
        self.model_id = Parser(plant = self.plant).AddModelFromFile(path, 'quad')

    @classmethod
    def createSimulator(cls, timestep, environment, controller):
        """
        Create a simulator object given the timestep, contact environment, and controller
        
        This is the main class method that should be used as follows:
            simulator, simbuilder = A1DrakeSimulationBuilder.createSimulator(timestep, environment, controller)

        From there, the user can get the plant and controller
        """
        this = cls(timestep)
        this._addEnvironment(environment)
        this._addController(controller)
        this._addVisualizer()
        this._compileDiagram()
        simulator = this._createSimulator()
        return simulator, this

    @staticmethod
    def get_a1_standing_state():
        """Create an initial state where A1 is standing"""
        a1 = A1()
        a1.Finalize()
        q0 = a1.standing_pose()
        q, status = a1.standing_pose_ik(q0[:7], guess=q0)
        if not status:
            print('IK Failed')
            return False
        else:
            v = np.zeros((a1.multibody.num_velocities(),))
            return np.concatenate([q, v], axis=0)

    def _addEnvironment(self, environment):
        """Add the contact environment to the diagram"""
        # Add the environment geometry
        self.plant = environment.addEnvironmentToPlant(self.plant)
        self.plant.Finalize()
        assert self.plant.geometry_source_is_registered()
        # Setup the scene graph
        self.builder.Connect(
            self.scene_graph.get_query_output_port(),
            self.plant.get_geometry_query_input_port()
        )
        self.builder.Connect(
            self.plant.get_geometry_poses_output_port(),
            self.scene_graph.get_source_pose_port(self.plant.get_source_id())
        )

    def _addController(self, controller_cls):
        """Add the controller and logger to the diagram"""
        # Add the controller to the system and wire within builder
        controller = controller_cls(self.plant, self.timestep)
        controller_sys = self.builder.AddSystem(controller)
        self.builder.Connect(
            controller_sys.GetOutputPort('torques'),
            self.plant.get_actuation_input_port(self.model_id)
        )
        self.builder.Connect(
            self.plant.get_state_output_port(),
            controller_sys.GetInputPort('state')
        )    
        # Add the controller logger
        self.logger = LogVectorOutput(controller_sys.GetOutputPort('logging'), self.builder)

    def _addVisualizer(self):
        """Add the Meshcat visualizer to the diagram"""
        self.meshcat = StartMeshcat()
        self.visualizer = ConnectMeshcatVisualizer(
            self.builder, 
            self.scene_graph, 
            zmq_url='default',
            open_browser=True,
            role = Role.kIllustration,
            prefix = 'visual'
            )
        self.collision_viz = ConnectMeshcatVisualizer(
            self.builder,
            self.scene_graph,
            zmq_url='default',
            open_browser=False,
            role = Role.kProximity,
            prefix = 'collision'
        )
        self.meshcat.SetProperty('collision','visible',False)

    def _compileDiagram(self):
        self.diagram = self.builder.Build()
        self.diagram.set_name('diagram')
        self.diagram_context = self.diagram.CreateDefaultContext()

    def _createSimulator(self):
        """Create and return the simulator object"""
        simulator = Simulator(self.diagram, self.diagram_context)
        simulator.set_target_realtime_rate(1.0)
        return simulator

    def set_initial_state(self, x0):
        """Set the initial state of the plant"""
        context = self.get_plant_context()
        self.plant.SetPositionsAndVelocities(context, x0)

    def get_plant(self):
        """Get the multibody plant object"""
        return self.plant
    
    def get_plant_context(self):
        """Get the mutable plant context from the diagram"""
        return self.diagram.GetMutableSubsystemContext(self.plant, self.diagram_context)

    def get_logs(self):
        """Return the data from the logger"""
        return self.logger.FindLog(self.diagram_context)

    def get_visualizer(self):
        return self.visualizer

class A1SimulationPlotter():
    """
    Plotting tool for a1 simulations
    """
    def __init__(self):
        self.a1 = A1()
        self.a1.Finalize()

    def plot(self, logs, show=True, savename=None):
        """Convert the logs to trajectories and plot"""
        # Grab the data
        t = logs.sample_times()
        nX = self.a1.multibody.num_positions() + self.a1.multibody.num_velocities()
        x, u = logs.data()[:nX, :], logs.data()[nX:, :]
        # Convert to trajectories
        xtraj = PiecewisePolynomial.FirstOrderHold(t, x)
        utraj = PiecewisePolynomial.ZeroOrderHold(t, u)
        # Plot the data
        self.a1.plot_trajectories(xtraj, utraj, show=show, savename=savename)

if __name__ == '__main__':
    print("Hello from a1_simulator")